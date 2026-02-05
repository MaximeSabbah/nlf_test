#!/usr/bin/env python3
import argparse, time, statistics as stats
import numpy as np
import cv2
import torch
from ultralytics import YOLO


def load_nlf_optimized(path: str, device: str):
    m = torch.jit.load(path).eval().to(device)

    def _nop(*args, **kwargs): return None
    try:
        m.forward = _nop
    except Exception:
        pass

    try:
        m = torch.jit.optimize_for_inference(
            m,
            other_methods=["estimate_poses_batched", "get_weights_for_canonical_points"],
        )
    except RuntimeError as e:
        print(f"[NLF] optimize_for_inference skipped: {e}")
    return m


def cast_fp16(x):
    if torch.is_tensor(x): return x.half()
    if isinstance(x, dict): return {k: cast_fp16(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(cast_fp16(v) for v in x)
    return x


def pct(vals, p):
    return float(np.percentile(np.array(vals, dtype=np.float64), p)) if vals else float("nan")


def summarize(name, vals):
    if not vals:
        print(f"{name}: (no data)")
        return
    print(
        f"{name}: mean={stats.mean(vals):.3f} ms | p50={pct(vals,50):.3f} | "
        f"p90={pct(vals,90):.3f} | p99={pct(vals,99):.3f} | n={len(vals)}"
    )


def make_intrinsics_from_fov(width, height, fov_deg=55.0):
    f = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    return np.array([[f, 0.0, width/2.0],
                     [0.0, f, height/2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def scale_intrinsics(K, sx, sy):
    K2 = K.copy()
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2


def build_boxes_xywh_score_top1(res, device="cuda:0"):
    # Always return either 0 boxes or 1 box (top confidence), to stabilize latency
    if len(res.boxes) == 0:
        return torch.zeros((0,5), device=device, dtype=torch.float32)

    conf = res.boxes.conf
    j = int(torch.argmax(conf).item())

    boxes = res.boxes.xyxy[j:j+1].to(device)             # (1,4)
    score = res.boxes.conf[j:j+1].to(device).unsqueeze(1) # (1,1)
    wh = boxes[:, 2:] - boxes[:, :2]
    out = torch.cat([boxes[:, :2], wh, score], dim=1).contiguous()
    return out.to(torch.float32)


def scale_boxes_xywh(boxes, sx, sy):
    if boxes.numel() == 0:
        return boxes
    b = boxes.clone()
    b[:,0] *= sx; b[:,2] *= sx
    b[:,1] *= sy; b[:,3] *= sy
    return b


def frame_to_nlf_tensor(frame_bgr, device, fp16, pinned_buf=None):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if pinned_buf is not None:
        pinned_buf.copy_(torch.from_numpy(rgb))
        cpu = pinned_buf
    else:
        cpu = torch.from_numpy(rgb)
    cpu = cpu.permute(2,0,1).contiguous()  # uint8 CHW
    x = cpu.to(device, non_blocking=(pinned_buf is not None))
    x = x.to(torch.float16 if fp16 else torch.float32).div_(255.0).unsqueeze(0)
    return x


def seek_caps(cap0, cap1, start):
    cap0.set(cv2.CAP_PROP_POS_FRAMES, start)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start)


def run_pass(mode, cap0, cap1, yolo, nlf0, nlf1, Kt, Et, world_up, weights,
             imgsz, conf, fp16, pinned0, pinned1, nlf_scale, start_frame,
             warmup, iters, discard):
    """
    mode: 'seq' or 'par'
    returns dict of timing lists.
    """
    device = "cuda:0"
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    ready_evt = torch.cuda.Event()

    # timers
    y_s = torch.cuda.Event(enable_timing=True); y_e = torch.cuda.Event(enable_timing=True)
    s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)

    # warmup (YOLO + NLF) — IMPORTANT
    seek_caps(cap0, cap1, start_frame)
    with torch.inference_mode():
        for _ in range(warmup):
            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()
            if not (ok0 and ok1):
                break
            _ = yolo.predict([f0, f1], imgsz=imgsz, classes=0, conf=conf, device=0, verbose=False)
            # minimal NLF warmup with empty box (stable shape)
            zbox = torch.zeros((0,5), device=device, dtype=torch.float32)
            img0 = frame_to_nlf_tensor(f0, device, fp16, pinned0)
            img1 = frame_to_nlf_tensor(f1, device, fp16, pinned1)
            _ = nlf0.estimate_poses_batched(img0, [zbox], intrinsic_matrix=Kt, extrinsic_matrix=Et,
                                            world_up_vector=world_up, weights=weights, num_aug=1)
            _ = nlf1.estimate_poses_batched(img1, [zbox], intrinsic_matrix=Kt, extrinsic_matrix=Et,
                                            world_up_vector=world_up, weights=weights, num_aug=1)
        torch.cuda.synchronize()

    # measure
    seek_caps(cap0, cap1, start_frame)
    out = {"decode_ms": [], "yolo_ms": [], "nlf_ms": [], "e2e_ms": []}

    s = float(nlf_scale)
    base_w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    base_h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nlf_w = int(round(base_w * s))
    nlf_h = int(round(base_h * s))
    sx = nlf_w / base_w
    sy = nlf_h / base_h

    measured = 0
    with torch.inference_mode():
        while measured < iters + discard:
            t_read0 = time.perf_counter()
            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()
            t_read1 = time.perf_counter()
            if not (ok0 and ok1):
                break
            decode_ms = (t_read1 - t_read0) * 1000.0

            # optional NLF downscale (offline -> acceptable)
            if s != 1.0:
                f0n = cv2.resize(f0, (nlf_w, nlf_h), interpolation=cv2.INTER_LINEAR)
                f1n = cv2.resize(f1, (nlf_w, nlf_h), interpolation=cv2.INTER_LINEAR)
            else:
                f0n, f1n = f0, f1

            # YOLO (batched)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            y_s.record()
            yres = yolo.predict([f0, f1], imgsz=imgsz, classes=0, conf=conf, device=0, verbose=False)
            y_e.record()

            # stabilize boxes: top-1 only
            b0 = build_boxes_xywh_score_top1(yres[0], device=device)
            b1 = build_boxes_xywh_score_top1(yres[1], device=device)
            if s != 1.0:
                b0 = scale_boxes_xywh(b0, sx, sy)
                b1 = scale_boxes_xywh(b1, sx, sy)

            # tensors (default stream)
            img0 = frame_to_nlf_tensor(f0n, device, fp16, pinned0 if (s == 1.0) else None)
            img1 = frame_to_nlf_tensor(f1n, device, fp16, pinned1 if (s == 1.0) else None)

            ready_evt.record()

            if mode == "seq":
                # sequential on default stream
                s0.record()
                _ = nlf0.estimate_poses_batched(img0, [b0], intrinsic_matrix=Kt, extrinsic_matrix=Et,
                                                world_up_vector=world_up, weights=weights, num_aug=1)
                e0.record()

                s1.record()
                _ = nlf1.estimate_poses_batched(img1, [b1], intrinsic_matrix=Kt, extrinsic_matrix=Et,
                                                world_up_vector=world_up, weights=weights, num_aug=1)
                e1.record()
            else:
                # parallel in two streams
                with torch.cuda.stream(stream0):
                    stream0.wait_event(ready_evt)
                    s0.record()
                    _ = nlf0.estimate_poses_batched(img0, [b0], intrinsic_matrix=Kt, extrinsic_matrix=Et,
                                                    world_up_vector=world_up, weights=weights, num_aug=1)
                    e0.record()
                with torch.cuda.stream(stream1):
                    stream1.wait_event(ready_evt)
                    s1.record()
                    _ = nlf1.estimate_poses_batched(img1, [b1], intrinsic_matrix=Kt, extrinsic_matrix=Et,
                                                    world_up_vector=world_up, weights=weights, num_aug=1)
                    e1.record()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            yolo_ms = y_s.elapsed_time(y_e)
            nlf0_ms = s0.elapsed_time(e0)
            nlf1_ms = s1.elapsed_time(e1)
            nlf_ms = (nlf0_ms + nlf1_ms) if mode == "seq" else max(nlf0_ms, nlf1_ms)
            e2e_ms = (t1 - t0) * 1000.0  # includes YOLO + prep + NLF + any sync cost

            if measured >= discard:
                out["decode_ms"].append(decode_ms)
                out["yolo_ms"].append(yolo_ms)
                out["nlf_ms"].append(nlf_ms)
                out["e2e_ms"].append(e2e_ms)

            measured += 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video0", required=True)
    ap.add_argument("--video1", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--nlf", required=True)
    ap.add_argument("--cano", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--discard", type=int, default=20)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--geom_fp32", action="store_true")
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--nlf_scale", type=float, default=1.0)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = "cuda:0"
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
    print("gpu:", torch.cuda.get_device_name(0))

    cap0 = cv2.VideoCapture(args.video0)
    cap1 = cv2.VideoCapture(args.video1)
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Could not open videos")
    w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w0,h0) != (w1,h1):
        raise RuntimeError("Video resolutions differ")
    print(f"Video resolution: {w0}x{h0}")

    print("Loading YOLO...")
    yolo = YOLO(args.yolo, task="detect")

    print("Loading NLF (2 instances)...")
    nlf0 = load_nlf_optimized(args.nlf, device)
    nlf1 = load_nlf_optimized(args.nlf, device)

    # weights
    cano = np.load(args.cano)
    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,
               5882, 8680, 8892, 8596, 5902, 8589, 5895, 8482, 5788, 8846, 8634]
    pts = torch.from_numpy(cano[indices]).float().to(device)
    with torch.inference_mode():
        weights = nlf0.get_weights_for_canonical_points(pts)
        if args.fp16:
            weights = cast_fp16(weights)

    # intrinsics/extrinsics (single camera model; OK for benchmarking)
    K = make_intrinsics_from_fov(w0, h0, args.fov)
    s = float(args.nlf_scale)
    K = scale_intrinsics(K, s, s) if s != 1.0 else K
    E = np.eye(4, dtype=np.float32)

    geom_dtype = torch.float32 if args.geom_fp32 else (torch.float16 if args.fp16 else torch.float32)
    Kt = torch.from_numpy(K).to(device=device, dtype=geom_dtype).unsqueeze(0)
    Et = torch.from_numpy(E).to(device=device, dtype=geom_dtype).unsqueeze(0)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=geom_dtype)

    pinned0 = pinned1 = None
    if args.pinned and s == 1.0:
        pinned0 = torch.empty((h0, w0, 3), dtype=torch.uint8, device="cpu", pin_memory=True)
        pinned1 = torch.empty((h0, w0, 3), dtype=torch.uint8, device="cpu", pin_memory=True)

    print("\n=== PASS 1: SEQUENTIAL NLF ===")
    seq = run_pass("seq", cap0, cap1, yolo, nlf0, nlf1, Kt, Et, world_up, weights,
                   args.imgsz, args.conf, args.fp16, pinned0, pinned1,
                   args.nlf_scale, args.start, args.warmup, args.iters, args.discard)

    print("\n=== PASS 2: PARALLEL NLF (2 CUDA streams) ===")
    par = run_pass("par", cap0, cap1, yolo, nlf0, nlf1, Kt, Et, world_up, weights,
                   args.imgsz, args.conf, args.fp16, pinned0, pinned1,
                   args.nlf_scale, args.start, args.warmup, args.iters, args.discard)

    cap0.release(); cap1.release()

    print("\n=== RESULTS (after discard) ===")
    summarize("Decode (cap.read pair)", seq["decode_ms"])
    summarize("YOLO (seq pass)", seq["yolo_ms"])
    summarize("NLF (seq: sum cam0+cam1)", seq["nlf_ms"])
    summarize("E2E (seq wall)", seq["e2e_ms"])

    print("")
    summarize("YOLO (par pass)", par["yolo_ms"])
    summarize("NLF (par: max(cam0,cam1))", par["nlf_ms"])
    summarize("E2E (par wall)", par["e2e_ms"])

    if seq["e2e_ms"] and par["e2e_ms"]:
        gain = np.mean(np.array(seq["e2e_ms"]) - np.array(par["e2e_ms"]))
        print(f"\nAverage E2E gain (seq - par): {gain:.3f} ms")
        if gain > 0.5:
            print("=> Streams help (you get real overlap).")
        else:
            print("=> Little overlap (GPU saturated or internal sync).")


if __name__ == "__main__":
    main()
