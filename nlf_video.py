import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

def load_nlf_optimized(path: str, device: str):
    model = torch.jit.load(path).eval().to(device)

    # Injecte un forward factice si absent (comme dans le repo NLF)
    def _nop(*args, **kwargs):
        return None

    # TorchScript n'a parfois pas forward -> on le rajoute pour satisfaire optimize_for_inference
    try:
        model.forward = _nop
    except Exception:
        pass  # si torch refuse l'assignation, on continue sans (on fera sans optimize_for_inference)

    # Optimiser les méthodes réellement utilisées
    try:
        model = torch.jit.optimize_for_inference(
            model,
            other_methods=[
                "estimate_poses_batched",
                "get_weights_for_canonical_points",
            ],
        )
    except RuntimeError as e:
        print(f"[NLF] optimize_for_inference skipped: {e}")

    return model


def process_video(video_path, yolo_model_path, nlf_model_path, output_path="output_video.mp4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Charger les modèles
    yolo = YOLO(yolo_model_path)
    # nlf = torch.jit.load(nlf_model_path).eval().to(device)
    nlf = load_nlf_optimized(nlf_model_path, device)

    cano_verts_full = np.load("weights/canonical_verts/smplx.npy")

    world_up_vector=torch.tensor([0.0, -1.0, 0.0]).to(device)

    # 2. Préparer les poids une seule fois (pour tous les sommets ici)
    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,5882,8680,8892,8596,5902,8589,5895
               ,8482,5788,8846,8634]
    
    selected_points = torch.from_numpy(cano_verts_full[indices]).float().to(device)
    all_points = torch.from_numpy(cano_verts_full).float().to(device)


    weights = nlf.get_weights_for_canonical_points(all_points)

    # 3. Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Infos vidéo
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurer le VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Paramètres Caméra (fixe pour la vidéo)
    focal_length = width / (2 * np.tan(np.deg2rad(55) / 2))
    intrinsics = torch.tensor([[[focal_length, 0, width/2], 
                                [0, focal_length, height/2], 
                                [0, 0, 1]]]).float().to(device)
    extrinsics = torch.eye(4).unsqueeze(0).to(device)

    print(f"Début du traitement : {total_frames} frames à traiter...")

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # --- Etape A: YOLO ---
            torch.cuda.synchronize()
            start_yolo = time.time()
            # verbose=False pour éviter de polluer la console
            # yolo_results = yolo.predict(frame_bgr, classes=0, conf=0.25, device=device, verbose=False)[0]
            yolo_results = yolo.predict(
                                        frame_bgr,
                                        imgsz=640,          # important si engine statique
                                        classes=0,
                                        conf=0.25,
                                        verbose=False
                                    )[0]
            

            if len(yolo_results.boxes) > 0:
                boxes  = yolo_results.boxes.xyxy.to(device)
                scores = yolo_results.boxes.conf.to(device).unsqueeze(1)
                wh = boxes[:, 2:] - boxes[:, :2]
                nlf_boxes = [torch.cat([boxes[:, :2], wh, scores], dim=1).contiguous()]
            else:
                nlf_boxes = [torch.zeros((0, 5), device=device)]

            # --- Etape B: NLF ---
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).contiguous().to(device, non_blocking=True)

            if device == "cuda":
                img_tensor = img_tensor.to(torch.float16).div_(255.0)
            else:
                img_tensor = img_tensor.to(torch.float32).div_(255.0)

            img_tensor = img_tensor.unsqueeze(0)
            torch.cuda.synchronize()

            yolo_time = (time.time() - start_yolo) * 1000  # Conversion en ms
            
            torch.cuda.synchronize()
            start_nlf = time.time()
            with torch.inference_mode():
                outputs = nlf.estimate_poses_batched(
                    img_tensor, nlf_boxes,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    world_up_vector=world_up_vector,
                    weights=weights,
                    num_aug=1
                )

            torch.cuda.synchronize()
            nlf_time = (time.time() - start_nlf) * 1000
            total_time = (time.time() - start_yolo) * 1000

            # --- Etape C: Dessin ---
            poses_3d = outputs['poses3d'][0]
            processed_frame = draw_projections(frame_bgr, poses_3d, intrinsics)
            
            
            # Afficher les infos sur la frame (optionnel)
            cv2.putText(processed_frame, f"NLF: {nlf_time:.1f}ms | Total: {total_time:.1f}ms", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Ecrire la frame
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 1 == 0:
                print("nlf_time",nlf_time)
                print("yolo_time",yolo_time)
                print(f"Frame {frame_count}/{total_frames} traitée ({total_time:.1f} ms/frame)")

    finally:
        cap.release()
        out.release()
        print(f"Traitement terminé. Vidéo sauvegardée sous : {output_path}")

def draw_projections(img_bgr, poses_3d, intrinsics):
    img_viz = img_bgr.copy()
    poses_3d_np = poses_3d.cpu().numpy()
    K = intrinsics[0].cpu().numpy()

    for person_id in range(poses_3d_np.shape[0]):
        points_3d = poses_3d_np[person_id]
        # On ajuste la taille des points selon le nombre (6890 vs 20)
        radius = 1 if len(points_3d) > 100 else 4
        color = (0, 255, 255) if len(points_3d) > 100 else (0, 0, 255)

        for pt_3d in points_3d:
            x_3d, y_3d, z_3d = pt_3d
            if z_3d != 0:
                u = (K[0, 0] * x_3d / z_3d) + K[0, 2]
                v = (K[1, 1] * y_3d / z_3d) + K[1, 2]
                cv2.circle(img_viz, (int(u), int(v)), radius, color, -1)
    return img_viz

# --- LANCEMENT ---
if __name__ == "__main__":
    video_in = "data/camera_0.mp4"
    yolo_p = "weights/yolo/yolov10n.pt"
    nlf_p = "weights/nlf/nlf_s_multi_0.2.2.torchscript"
    
    process_video(video_in, yolo_p, nlf_p, "resultat_final.mp4")
