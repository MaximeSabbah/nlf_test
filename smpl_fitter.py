#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NLF (+ YOLO) -> (optionnel) fit SMPL/SMPL-X -> vidéo de debug

Dépendances (indicatif) :
  pip install ultralytics opencv-python torch numpy smplfitter

Exemple :
  python3 nlf_fit_smpl_video.py \
    --video data/camera_0.mp4 \
    --yolo yolo11m.pt \
    --nlf weights/nlf/nlf_s_multi_0.2.2.torchscript \
    --canonical-verts canonical_verts/smplx.npy \
    --out resultat_final.mp4 \
    --fit-smpl \
    --fit-device cuda \
    --draw joints
"""

import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import smplfitter.pt


# --------------------------
# Utils
# --------------------------

def cuda_sync_if_needed(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def compute_intrinsics_from_fov(width: int, height: int, fov_deg: float, device: str) -> torch.Tensor:
    # FOV horizontal (approx)
    focal_length = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    K = torch.tensor(
        [[
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ]],
        dtype=torch.float32,
        device=device,
    )
    return K


def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    points_3d: (N,3)
    K: (3,3)
    return: (N,2) float (u,v)
    """
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    # éviter division par 0
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    return np.stack([u, v], axis=1)


def draw_points(img_bgr: np.ndarray, points_3d: torch.Tensor, intrinsics: torch.Tensor,
                radius: int = 3, color=(0, 0, 255), stride: int = 1) -> np.ndarray:
    """
    points_3d: (N,3) torch
    intrinsics: (1,3,3) torch
    """
    img = img_bgr.copy()
    pts = points_3d.detach().float().cpu().numpy()
    K = intrinsics[0].detach().cpu().numpy()

    pts = pts[::stride]
    uv = project_points(pts, K)

    h, w = img.shape[:2]
    for u, v in uv:
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(img, (ui, vi), radius, color, -1)
    return img


def infer_unit_scale(points_3d: torch.Tensor) -> float:
    """
    Heuristique : si les profondeurs sont plutôt > 10 (typiquement mm), on convertit en mètres.
    """
    with torch.no_grad():
        z = points_3d[..., 2].detach()
        z = z[torch.isfinite(z)]
        if z.numel() == 0:
            return 1.0
        med = torch.median(torch.abs(z)).item()
    # med ~ 1500 -> mm, med ~ 1.5 -> m
    return 1.0 / 1000.0 if med > 10.0 else 1.0


def make_vertex_subset(V: int, n_subset: int) -> Tuple[int, ...]:
    if n_subset >= V:
        return tuple(range(V))
    idx = np.linspace(0, V - 1, n_subset, dtype=np.int64)
    return tuple(idx.tolist())


# --------------------------
# NLF loading
# --------------------------

def load_nlf_optimized(path: str, device: str, optimize: bool = True) -> torch.jit.ScriptModule:
    model = torch.jit.load(path).eval().to(device)

    # Hack: certains TorchScript n'ont pas forward, optimize_for_inference peut râler
    def _nop(*args, **kwargs):
        return None

    try:
        model.forward = _nop
    except Exception:
        pass

    if optimize:
        try:
            model = torch.jit.optimize_for_inference(
                model,
                other_methods=[
                    "estimate_poses_batched",
                    "get_weights_for_canonical_points",
                ],
            )
        except Exception as e:
            print(f"[NLF] optimize_for_inference skipped: {e}")

    return model


# --------------------------
# SMPL/SMPL-X fitter wrapper
# --------------------------

class SMPLFitter:
    def __init__(self, body_model_name: str, device: str, n_subset: int = 1024, num_betas: int = 10):
        self.body_model_name = body_model_name
        self.device = device
        self.bm = smplfitter.pt.get_cached_body_model(body_model_name)
        self.V = self.bm.num_vertices
        self.J = self.bm.num_joints

        self.vertex_subset = make_vertex_subset(self.V, n_subset=n_subset)
        self.vertex_subset_pt = torch.tensor(self.vertex_subset, device=self.device, dtype=torch.int64)

        # Fit fn (créée une seule fois)
        self.fit_fn = smplfitter.pt.get_cached_fit_fn(
            body_model_name=self.body_model_name,
            requested_keys=("pose_rotvecs", "shape_betas", "trans"),
            num_betas=num_betas,
            vertex_subset=self.vertex_subset,
            share_beta=False,          # fit frame-by-frame
            final_adjust_rots=True,
            device=self.device,
            beta_regularizer=10.0,
            beta_regularizer2=0.2,
        )

    def fit(self,
            vertices: torch.Tensor,  # (N,V,3) float32
            joints: torch.Tensor,    # (N,J,3) float32 (peut être vide)
            v_w: torch.Tensor,       # (N,V) float32
            j_w: torch.Tensor        # (N,J) float32 (peut être vide)
            ):
        # subset vertices
        vertices_sub = torch.index_select(vertices, dim=1, index=self.vertex_subset_pt)
        v_w_sub = torch.index_select(v_w, dim=1, index=self.vertex_subset_pt)

        fits = self.fit_fn(vertices_sub, joints, v_w_sub, j_w)

        # reconstruire vertices/joints propres
        fit_res = self.bm.forward(fits["pose_rotvecs"], fits["shape_betas"], fits["trans"])
        return fits, fit_res  # fit_res["vertices"], fit_res["joints"]


# --------------------------
# Main pipeline
# --------------------------

def process_video(
    video_path: str,
    yolo_model_path: str,
    nlf_model_path: str,
    canonical_verts_path: str,
    output_path: str,
    fov_deg: float,
    imgsz: int,
    conf: float,
    device: str,
    nlf_fp16: bool,
    optimize_nlf: bool,
    fit_smpl: bool,
    fit_device: str,
    fit_every: int,
    draw_mode: str,
    draw_stride: int,
):
    # Models
    yolo = YOLO(yolo_model_path)
    nlf = load_nlf_optimized(nlf_model_path, device=device, optimize=optimize_nlf)

    # Canonical verts -> weights (NLF)
    cano_verts_full = np.load(canonical_verts_path)  # (Vcano,3)
    all_points = torch.from_numpy(cano_verts_full).float().to(device)
    with torch.inference_mode():
        weights = nlf.get_weights_for_canonical_points(all_points)

    world_up_vector = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)

    # Video IO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Camera (fixe)
    intrinsics = compute_intrinsics_from_fov(width, height, fov_deg, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)

    # SMPL fitter (lazy init)
    fitter: Optional[SMPLFitter] = None
    body_model_name: Optional[str] = None
    unit_scale: Optional[float] = None

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Resolution: {width}x{height} | fps={fps:.2f} | frames={total_frames}")
    print(f"[INFO] Device: {device} | NLF fp16={nlf_fp16} | fit_smpl={fit_smpl} (fit_device={fit_device})")

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            t0 = time.time()

            # ---------------- YOLO ----------------
            cuda_sync_if_needed(device)
            t_y0 = time.time()

            # ultralytics: device=0 (GPU0) ou "cpu"
            yolo_dev = 0 if device.startswith("cuda") else "cpu"
            yolo_res = yolo.predict(
                frame_bgr,
                imgsz=imgsz,
                classes=0,
                conf=conf,
                device=yolo_dev,
                verbose=False,
            )[0]

            if len(yolo_res.boxes) > 0:
                boxes_xyxy = yolo_res.boxes.xyxy
                scores = yolo_res.boxes.conf

                # s'assurer sur torch + device
                boxes_xyxy = boxes_xyxy.to(device)
                scores = scores.to(device).unsqueeze(1)

                wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
                boxes_xywhs = torch.cat([boxes_xyxy[:, :2], wh, scores], dim=1).contiguous()
                nlf_boxes = [boxes_xywhs]
            else:
                nlf_boxes = [torch.zeros((0, 5), device=device, dtype=torch.float32)]

            cuda_sync_if_needed(device)
            yolo_time = (time.time() - t_y0) * 1000.0

            # ---------------- NLF ----------------
            cuda_sync_if_needed(device)
            t_n0 = time.time()

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()

            # vers GPU
            img_tensor = img_tensor.to(device, non_blocking=True)

            # normalisation + dtype
            if device.startswith("cuda") and nlf_fp16:
                img_tensor = img_tensor.to(torch.float16).div_(255.0)
            else:
                img_tensor = img_tensor.to(torch.float32).div_(255.0)

            img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)

            with torch.inference_mode():
                outputs = nlf.estimate_poses_batched(
                    img_tensor,
                    nlf_boxes,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    world_up_vector=world_up_vector,
                    weights=weights,
                    num_aug=1,
                )

            cuda_sync_if_needed(device)
            nlf_time = (time.time() - t_n0) * 1000.0

            # outputs["poses3d"] est généralement une liste (batch) -> [0]
            poses3d = outputs["poses3d"][0]  # (Npeople, Npoints, 3) torch

            # si aucune personne, on écrit juste la frame + overlay temps
            if poses3d.numel() == 0 or poses3d.shape[0] == 0:
                total_time = (time.time() - t0) * 1000.0
                frame_out = frame_bgr.copy()
                cv2.putText(frame_out, f"YOLO {yolo_time:.1f}ms | NLF {nlf_time:.1f}ms | TOTAL {total_time:.1f}ms",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                out.write(frame_out)
                frame_idx += 1
                continue

            # init unit scale au premier frame utile
            if unit_scale is None:
                unit_scale = infer_unit_scale(poses3d)
                if abs(unit_scale - 1.0) > 1e-6:
                    print(f"[INFO] Unit scale inferred: {unit_scale} (conversion vers mètres)")

            # ---------------- SMPL FIT (optionnel) ----------------
            fit_time = 0.0
            smpl_vertices = None
            smpl_joints = None

            if fit_smpl and (frame_idx % max(1, fit_every) == 0):
                cuda_sync_if_needed(fit_device)
                t_f0 = time.time()

                # Lazy init fitter : on choisit smpl ou smplx en regardant Npoints
                Npeople, Npoints, _ = poses3d.shape

                if fitter is None:
                    # Heuristique “simple”: si Npoints ~ 10475 -> smplx, si ~ 6890 -> smpl
                    # Sinon on essaye smplx puis smpl.
                    candidates = []
                    if 10300 <= Npoints <= 11000:
                        candidates = ["smplx", "smpl"]
                    elif 6700 <= Npoints <= 7200:
                        candidates = ["smpl", "smplx"]
                    else:
                        candidates = ["smplx", "smpl"]

                    last_err = None
                    for cand in candidates:
                        try:
                            tmp = SMPLFitter(cand, device=fit_device, n_subset=1024, num_betas=10)
                            V, J = tmp.V, tmp.J
                            if Npoints == V or Npoints == V + J or Npoints == J:
                                fitter = tmp
                                body_model_name = cand
                                print(f"[INFO] SMPL fitter initialized: {cand} (V={V}, J={J}), Npoints={Npoints}")
                                break
                        except Exception as e:
                            last_err = e
                            continue

                    if fitter is None:
                        raise RuntimeError(
                            f"Impossible d'initialiser un fitter SMPL/SMPL-X: Npoints={Npoints}. "
                            f"Dernière erreur: {last_err}"
                        )

                assert fitter is not None
                V, J = fitter.V, fitter.J

                # Split vertices/joints selon le format
                poses3d_f = poses3d.detach()

                # Conversion en mètres si besoin
                poses3d_f = poses3d_f.float() * float(unit_scale)

                if Npoints == V:
                    vertices = poses3d_f.to(fit_device)
                    joints = torch.empty((Npeople, 0, 3), device=fit_device, dtype=torch.float32)
                    v_w = torch.ones((Npeople, V), device=fit_device, dtype=torch.float32)
                    j_w = torch.empty((Npeople, 0), device=fit_device, dtype=torch.float32)

                elif Npoints == V + J:
                    vertices = poses3d_f[:, :V].to(fit_device)
                    joints = poses3d_f[:, V:V + J].to(fit_device)
                    v_w = torch.ones((Npeople, V), device=fit_device, dtype=torch.float32)
                    j_w = torch.ones((Npeople, J), device=fit_device, dtype=torch.float32)

                elif Npoints == J:
                    # joints only (pas idéal mais ça peut marcher)
                    vertices = torch.zeros((Npeople, V, 3), device=fit_device, dtype=torch.float32)
                    joints = poses3d_f.to(fit_device)
                    v_w = torch.zeros((Npeople, V), device=fit_device, dtype=torch.float32)
                    j_w = torch.ones((Npeople, J), device=fit_device, dtype=torch.float32)

                else:
                    raise RuntimeError(f"Format poses3d inattendu: Npoints={Npoints} vs V={V}, J={J}")

                # Fit
                fits, fit_res = fitter.fit(vertices, joints, v_w, j_w)
                smpl_vertices = fit_res["vertices"]  # (Npeople,V,3)
                smpl_joints = fit_res["joints"]      # (Npeople,J?,3)

                cuda_sync_if_needed(fit_device)
                fit_time = (time.time() - t_f0) * 1000.0

            # ---------------- Draw ----------------
            frame_out = frame_bgr.copy()

            # par défaut, on dessine les joints du SMPL fit si dispo, sinon les points NLF
            if fit_smpl and smpl_joints is not None:
                for pid in range(smpl_joints.shape[0]):
                    pts = smpl_joints[pid].to(device)  # pour projeter avec K sur device
                    frame_out = draw_points(
                        frame_out,
                        pts,
                        intrinsics,
                        radius=4,
                        color=(0, 0, 255),
                        stride=1,
                    )
            else:
                # NLF raw : dessiner soit joints soit verts subsamplés
                # On ne sait pas a priori où sont les joints, donc:
                # - si beaucoup de points, on subsample
                for pid in range(poses3d.shape[0]):
                    pts = poses3d[pid]
                    # conversion m si unit_scale != 1
                    pts = pts.float() * float(unit_scale if unit_scale is not None else 1.0)
                    if draw_mode == "joints":
                        # si très dense, on prend seulement 24/55 premiers points (approx) -> fallback
                        n = pts.shape[0]
                        n_keep = min(32, n)
                        pts_draw = pts[:n_keep]
                        frame_out = draw_points(frame_out, pts_draw, intrinsics, radius=4, color=(0, 255, 255), stride=1)
                    else:
                        stride = max(1, draw_stride)
                        frame_out = draw_points(frame_out, pts, intrinsics, radius=2, color=(0, 255, 255), stride=stride)

            total_time = (time.time() - t0) * 1000.0
            overlay = f"YOLO {yolo_time:.1f}ms | NLF {nlf_time:.1f}ms"
            if fit_smpl:
                overlay += f" | FIT {fit_time:.1f}ms"
            overlay += f" | TOTAL {total_time:.1f}ms"
            cv2.putText(frame_out, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame_out)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"[{frame_idx:05d}] {overlay}")

    finally:
        cap.release()
        out.release()
        print(f"[DONE] Vidéo sauvegardée : {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=str)
    p.add_argument("--yolo", required=True, type=str)
    p.add_argument("--nlf", required=True, type=str)
    p.add_argument("--canonical-verts", required=True, type=str)
    p.add_argument("--out", default="output_video.mp4", type=str)

    p.add_argument("--fov", default=55.0, type=float)
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--conf", default=0.25, type=float)

    p.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    p.add_argument("--nlf-fp16", action="store_true", help="Entrée NLF en float16 (GPU)")
    p.add_argument("--no-nlf-opt", action="store_true", help="Désactive torch.jit.optimize_for_inference")

    p.add_argument("--fit-smpl", action="store_true", help="Active le fit SMPL/SMPL-X (frame par frame)")
    p.add_argument("--fit-device", default="cuda", type=str, choices=["cuda", "cpu"])
    p.add_argument("--fit-every", default=1, type=int, help="Fit 1 frame sur N (pour gagner du temps)")

    p.add_argument("--draw", default="joints", choices=["joints", "verts"], help="Mode de dessin si pas de fit")
    p.add_argument("--draw-stride", default=10, type=int, help="Sous-échantillonnage des points en mode verts")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    fit_device = "cuda" if (args.fit_device == "cuda" and torch.cuda.is_available()) else "cpu"

    process_video(
        video_path=args.video,
        yolo_model_path=args.yolo,
        nlf_model_path=args.nlf,
        canonical_verts_path=args.canonical_verts,
        output_path=args.out,
        fov_deg=args.fov,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        nlf_fp16=bool(args.nlf_fp16),
        optimize_nlf=not bool(args.no_nlf_opt),
        fit_smpl=bool(args.fit_smpl),
        fit_device=fit_device,
        fit_every=int(args.fit_every),
        draw_mode=str(args.draw),
        draw_stride=int(args.draw_stride),
    )
