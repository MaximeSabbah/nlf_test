#!/usr/bin/env python3
import os
import sys
import time
import argparse
import statistics as stats
from collections import defaultdict

import numpy as np
import cv2
import torch
from ultralytics import YOLO


# ----------------------------
# Helpers: timing
# ----------------------------
def ms_cpu(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out


def ms_gpu(fn):
    # Accurate GPU section timing (CUDA events)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    out = fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender), out


def percentile(vals, p):
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def summarize(name, vals):
    if not vals:
        print(f"{name}: (no data)")
        return
    print(
        f"{name}: mean={stats.mean(vals):.3f} ms | "
        f"p50={percentile(vals,50):.3f} | p90={percentile(vals,90):.3f} | "
        f"p99={percentile(vals,99):.3f} | n={len(vals)}"
    )


# ----------------------------
# Environment diagnostics
# ----------------------------
def print_env():
    import platform
    print("=== ENV ===")
    print("platform:", platform.platform())
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
    print("opencv:", cv2.__version__)
    print("numpy:", np.__version__)
    # threads
    try:
        print("cv2 threads:", cv2.getNumThreads())
    except Exception:
        pass
    print("torch num_threads:", torch.get_num_threads())
    # affinity
    try:
        aff = os.sched_getaffinity(0)
        print("cpu affinity count:", len(aff))
    except Exception:
        print("cpu affinity: (unavailable)")
    # /dev/shm
    try:
        st = os.statvfs("/dev/shm")
        shm_gb = (st.f_frsize * st.f_blocks) / (1024**3)
        print(f"/dev/shm size: {shm_gb:.2f} GiB")
    except Exception:
        print("/dev/shm size: (unavailable)")
    # cgroup cpu throttling hint (cgroup v2)
    for p in ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    print(p + ":", f.read().strip())
            except Exception:
                pass
    print("env OMP/MKL:", {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
    print("============\n")


# ----------------------------
# Core benchmark
# ----------------------------
def build_boxes_from_yolo(yolo_results, device):
    # returns list with one tensor (n,5) in xywh + score
    if len(yolo_results.boxes) > 0:
        boxes = yolo_results.boxes.xyxy  # (n,4) on GPU
        scores = yolo_results.boxes.conf.unsqueeze(1)  # (n,1) on GPU
        wh = boxes[:, 2:] - boxes[:, :2]
        boxes_formatted = torch.cat([boxes[:, :2], wh, scores], dim=1)
        return [boxes_formatted]
    else:
        return [torch.zeros((0, 5), device=device)]


def prepare_intrinsics(width, height, fov_deg, device):
    focal_length = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    intrinsics = torch.tensor([[[focal_length, 0, width / 2.0],
                                [0, focal_length, height / 2.0],
                                [0, 0, 1]]], dtype=torch.float32, device=device)
    extrinsics = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
    return intrinsics, extrinsics, world_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--yolo", required=True, help="Path to YOLO weights (e.g., yolo11m.pt)")
    ap.add_argument("--nlf", required=True, help="Path to NLF TorchScript (e.g., nlf_*.torchscript)")
    ap.add_argument("--cano", required=True, help="Path to canonical verts npy (e.g., canonical_verts/smplx.npy)")
    ap.add_argument("--fov", type=float, default=55.0, help="Camera horizontal FOV in degrees")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200, help="Number of measured frames (after warmup)")
    ap.add_argument("--start", type=int, default=0, help="Start frame index")
    ap.add_argument("--use_pinned", action="store_true", help="Use pinned memory for H2D copy")
    ap.add_argument("--cv_threads", type=int, default=None, help="Force OpenCV threads (e.g., 1)")
    ap.add_argument("--torch_threads", type=int, default=None, help="Force torch CPU threads (e.g., 1)")
    args = ap.parse_args()

    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    print_env()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark is intended for GPU runs.")
        sys.exit(1)

    device = "cuda:0"

    # Load models
    print("Loading YOLO...")
    yolo = YOLO(args.yolo)
    print("Loading NLF TorchScript...")
    nlf = torch.jit.load(args.nlf).eval().to(device)

    # Canonical verts + weights
    cano_verts_full = np.load(args.cano)
    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,
               5882, 8680, 8892, 8596, 5902, 8589, 5895, 8482, 5788, 8846, 8634]
    selected_points = torch.from_numpy(cano_verts_full[indices]).float().to(device)

    # (Optional) also keep full points if you need later
    # all_points = torch.from_numpy(cano_verts_full).float().to(device)

    # Warm compute of weights (GPU timed)
    w_ms, weights = ms_gpu(lambda: nlf.get_weights_for_canonical_points(selected_points))
    print(f"weights precompute (GPU): {w_ms:.3f} ms\n")

    # Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.2f} fps | frames={total}")

    intrinsics, extrinsics, world_up = prepare_intrinsics(width, height, args.fov, device)

    # Seek to start
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # Storage
    times = defaultdict(list)

    # Warmup + measured loop
    n_total = args.warmup + args.iters
    read_frames = 0

    # Pre-allocate pinned buffer optionally
    pinned = None
    if args.use_pinned:
        pinned = torch.empty((height, width, 3), dtype=torch.uint8, device="cpu", pin_memory=True)

    print(f"\nRunning: warmup={args.warmup} | measured={args.iters} | pinned={args.use_pinned}\n")

    with torch.inference_mode():
        while read_frames < n_total:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # CPU: BGR->RGB
            t_cvt, img_rgb = ms_cpu(lambda: cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            # CPU: numpy->torch (CPU tensor) + permute/contiguous
            def cpu_tensorize():
                # uint8 HWC -> uint8 HWC tensor
                if args.use_pinned:
                    pinned.copy_(torch.from_numpy(img_rgb))  # copy into pinned
                    t = pinned
                else:
                    t = torch.from_numpy(img_rgb)
                t = t.permute(2, 0, 1).contiguous()  # CHW uint8
                return t
            t_torch_cpu, img_cpu = ms_cpu(cpu_tensorize)

            # GPU: H2D + float/normalize + batch
            def h2d_and_norm():
                x = img_cpu.to(device, non_blocking=args.use_pinned)
                x = x.float().div_(255.0).unsqueeze(0)  # 1x3xHxW
                return x
            t_h2d, img_tensor = ms_gpu(h2d_and_norm)

            # GPU: YOLO predict (timed with CUDA events)
            t_yolo, yolo_res = ms_gpu(lambda: yolo.predict(frame_bgr, classes=0, conf=0.25, device=0, verbose=False)[0])

            # GPU: box formatting (small, but time it)
            t_box, nlf_boxes = ms_gpu(lambda: build_boxes_from_yolo(yolo_res, device))

            # GPU: NLF forward
            def nlf_forward():
                return nlf.estimate_poses_batched(
                    img_tensor, nlf_boxes,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    world_up_vector=world_up,
                    weights=weights,
                    num_aug=1,
                )
            t_nlf, outputs = ms_gpu(nlf_forward)

            # GPU: optional sync point already included in ms_gpu; compute total GPU-ish time
            # CPU: total frame wall
            # For total wall, just measure outer loop with perf_counter around everything (we can approximate using sum too)
            # Here we compute a pseudo-total as sum of measured components:
            t_total = t_cvt + t_torch_cpu + t_h2d + t_yolo + t_box + t_nlf

            if read_frames >= args.warmup:
                times["cpu_cvt_ms"].append(t_cvt)
                times["cpu_tensorize_ms"].append(t_torch_cpu)
                times["gpu_h2d_norm_ms"].append(t_h2d)
                times["gpu_yolo_ms"].append(t_yolo)
                times["gpu_box_ms"].append(t_box)
                times["gpu_nlf_ms"].append(t_nlf)
                times["sum_components_ms"].append(t_total)

            read_frames += 1

            if read_frames in [args.warmup, args.warmup + 50, args.warmup + 100, args.warmup + args.iters]:
                done = max(0, read_frames - args.warmup)
                print(f"progress: {done}/{args.iters}")

    cap.release()

    print("\n=== RESULTS (measured frames only) ===")
    summarize("CPU cvtColor", times["cpu_cvt_ms"])
    summarize("CPU tensorize (numpy->torch CPU)", times["cpu_tensorize_ms"])
    summarize("GPU H2D+float+normalize", times["gpu_h2d_norm_ms"])
    summarize("GPU YOLO predict", times["gpu_yolo_ms"])
    summarize("GPU box format", times["gpu_box_ms"])
    summarize("GPU NLF estimate", times["gpu_nlf_ms"])
    summarize("Sum of components (approx total)", times["sum_components_ms"])

    print("\nTip: If Docker differs mainly in CPU parts, suspect CPU quotas/affinity, OpenCV threads, /dev/shm, filesystem.")
    print("Tip: If Docker differs mainly in GPU parts, check nvidia-smi clocks/P-state and GPU contention.\n")


if __name__ == "__main__":
    main()
