#!/usr/bin/env python3
"""
Benchmark YOLO (TensorRT via Ultralytics) + NLF (TorchScript) for N cameras (2 or 4).

- Measures GPU time with CUDA events (accurate).
- Measures CPU preprocessing separately (cvtColor + tensorize).
- Supports:
    * --cams 2 or 4
    * --fp16 (for NLF inputs + weights; YOLO engine already defines precision)
    * --pinned (pinned host buffers for faster H2D)
    * --warmup / --iters / --start
- No drawing, no video writing (keeps benchmark honest).

Example:
  python bench_multi_cam.py \
    --cams 4 \
    --videos data/camera_0.mp4 data/camera_1.mp4 data/camera_2.mp4 data/camera_3.mp4 \
    --yolo yolo11m.engine \
    --nlf weights/nlf/nlf_s_multi_0.2.2.torchscript \
    --cano canonical_verts/smplx.npy \
    --imgsz 640 --conf 0.25 --warmup 30 --iters 200 --fp16 --pinned
"""
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
# Timing helpers
# ----------------------------
def ms_cpu(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out


def ms_gpu(fn):
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
# Model utils
# ----------------------------
def load_nlf_optimized(path: str, device: str):
    model = torch.jit.load(path).eval().to(device)

    def _nop(*args, **kwargs):
        return None

    try:
        model.forward = _nop
    except Exception:
        pass

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


def cast_fp16(x):
    if torch.is_tensor(x):
        return x.half()
    if isinstance(x, dict):
        return {k: cast_fp16(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cast_fp16(v) for v in x)
    return x


def build_boxes_list_from_ultralytics(results, device):
    """
    results: list[ultralytics.engine.results.Results], length B
    Returns: list length B, each tensor (Ni,5) containing [x,y,w,h,score] on GPU
    """
    out = []
    for res in results:
        if len(res.boxes) > 0:
            boxes = res.boxes.xyxy.to(device)
            scores = res.boxes.conf.to(device).unsqueeze(1)
            wh = boxes[:, 2:] - boxes[:, :2]
            out.append(torch.cat([boxes[:, :2], wh, scores], dim=1).contiguous())
        else:
            out.append(torch.zeros((0, 5), device=device))
    return out


def make_intrinsics_batch(widths, heights, fov_deg, device, dtype):
    """
    widths/heights: list length B
    returns intrinsics (B,3,3) and extrinsics (B,4,4)
    """
    Ks = []
    for w, h in zip(widths, heights):
        f = w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
        K = torch.tensor([[f, 0.0, w / 2.0],
                          [0.0, f, h / 2.0],
                          [0.0, 0.0, 1.0]], dtype=dtype, device=device)
        Ks.append(K)
    intrinsics = torch.stack(Ks, dim=0)  # (B,3,3)
    extrinsics = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(len(widths), 1, 1)
    return intrinsics, extrinsics


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
    try:
        print("cv2 threads:", cv2.getNumThreads())
    except Exception:
        pass
    print("torch num_threads:", torch.get_num_threads())
    try:
        aff = os.sched_getaffinity(0)
        print("cpu affinity count:", len(aff))
    except Exception:
        print("cpu affinity: (unavailable)")
    try:
        st = os.statvfs("/dev/shm")
        shm_gb = (st.f_frsize * st.f_blocks) / (1024**3)
        print(f"/dev/shm size: {shm_gb:.2f} GiB")
    except Exception:
        print("/dev/shm size: (unavailable)")
    if os.path.exists("/sys/fs/cgroup/cpu.max"):
        try:
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                print("/sys/fs/cgroup/cpu.max:", f.read().strip())
        except Exception:
            pass
    print("env OMP/MKL:", {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
    print("============\n")


# ----------------------------
# Core benchmark
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, choices=[2, 4], required=True, help="Number of cameras (2 or 4)")
    ap.add_argument("--videos", nargs="+", required=True, help="Video paths, length must match --cams")
    ap.add_argument("--yolo", required=True, help="YOLO model path (.engine or .pt)")
    ap.add_argument("--nlf", required=True, help="NLF TorchScript path")
    ap.add_argument("--cano", required=True, help="canonical verts npy path (canonical_verts/smplx.npy)")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz (keep consistent with engine)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--fp16", action="store_true", help="Use FP16 for NLF input + weights")
    ap.add_argument("--pinned", action="store_true", help="Use pinned host buffers for H2D")
    ap.add_argument("--cv_threads", type=int, default=None, help="Force OpenCV thread count (e.g., 1)")
    ap.add_argument("--torch_threads", type=int, default=None, help="Force torch CPU thread count (e.g., 1)")
    args = ap.parse_args()

    if len(args.videos) != args.cams:
        raise SystemExit(f"--videos must have exactly {args.cams} paths")

    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    torch.backends.cudnn.benchmark = True

    print_env()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available, expected GPU benchmark")

    device = "cuda:0"
    dtype = torch.float16 if args.fp16 else torch.float32

    # Load models
    print("Loading YOLO...")
    yolo = YOLO(args.yolo)  # .engine supported by Ultralytics
    print("Loading NLF TorchScript...")
    nlf = load_nlf_optimized(args.nlf, device)

    # Canonical verts / weights
    cano = np.load(args.cano)
    indices = [5621, 6629, 3878, 7040, 4302, 7105, 4369, 7584, 4848, 7457,
               4721, 8421, 5727, 8371, 5677, 6401, 3640, 6407, 3646, 8576,
               5882, 8680, 8892, 8596, 5902, 8589, 5895, 8482, 5788, 8846, 8634]
    selected_points = torch.from_numpy(cano[indices]).float().to(device)

    print("Precomputing weights...")
    w_ms, weights = ms_gpu(lambda: nlf.get_weights_for_canonical_points(selected_points))
    # if args.fp16:
    #     weights = cast_fp16(weights)
    print(f"weights precompute (GPU): {w_ms:.3f} ms\n")

    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=dtype, device=device)

    # Open video sources
    caps = []
    widths, heights, fpss, totals = [], [], [], []
    for vp in args.videos:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {vp}")
        if args.start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        caps.append(cap)
        widths.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        heights.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fpss.append(cap.get(cv2.CAP_PROP_FPS))
        totals.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Require same resolution for clean batching (recommended)
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise RuntimeError(f"All cameras must have same resolution for batching. widths={widths}, heights={heights}")

    width, height = widths[0], heights[0]
    print(f"Video: {width}x{height} | cams={args.cams} | fps={fpss} | frames={totals}")

    intrinsics, extrinsics = make_intrinsics_batch(widths, heights, args.fov, device, dtype)

    # Optional pinned buffers (one per cam)
    pinned_buffers = None
    if args.pinned:
        pinned_buffers = [
            torch.empty((height, width, 3), dtype=torch.uint8, device="cpu", pin_memory=True)
            for _ in range(args.cams)
        ]

    # Timing storage
    T = defaultdict(list)

    n_total = args.warmup + args.iters
    read_steps = 0

    print(f"\nRunning: cams={args.cams} | warmup={args.warmup} | measured={args.iters} | fp16={args.fp16} | pinned={args.pinned}\n")

    with torch.inference_mode():
        while read_steps < n_total:
            # 1) Read one frame per cam (CPU decode not included in fine breakdown unless you time it)
            frames_bgr = []
            ok_all = True
            for cap in caps:
                ok, frame = cap.read()
                if not ok:
                    ok_all = False
                    break
                frames_bgr.append(frame)
            if not ok_all:
                break

            # 2) CPU preprocessing: cvtColor + numpy->torch uint8 CHW (per cam), then stack
            def cpu_preproc():
                cpu_imgs = []
                for i, bgr in enumerate(frames_bgr):
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if args.pinned:
                        pinned_buffers[i].copy_(torch.from_numpy(rgb))
                        t = pinned_buffers[i]
                    else:
                        t = torch.from_numpy(rgb)
                    t = t.permute(2, 0, 1).contiguous()  # uint8 CHW on CPU
                    cpu_imgs.append(t)
                return cpu_imgs

            t_cpu_pre, cpu_imgs = ms_cpu(cpu_preproc)

            # 3) GPU: H2D + normalize + dtype + batch stack
            def h2d_norm():
                # stack on CPU then move once
                x = torch.stack(cpu_imgs, dim=0)  # (B,3,H,W) uint8 CPU
                x = x.to(device, non_blocking=args.pinned)
                if args.fp16:
                    x = x.to(torch.float16).div_(255.0)
                else:
                    x = x.to(torch.float32).div_(255.0)
                return x

            t_h2d, img_tensor = ms_gpu(h2d_norm)

            # 4) YOLO (batched) GPU timing
            def yolo_call():
                # Ultralytics accepts list of np arrays for batch predict
                # With TensorRT engine, keep imgsz fixed (engine often static)
                return yolo.predict(
                    frames_bgr,
                    imgsz=args.imgsz,
                    classes=0,
                    conf=args.conf,
                    device=0,
                    verbose=False,
                )

            t_yolo, yolo_results = ms_gpu(yolo_call)

            # 5) Build NLF boxes list (GPU, small)
            t_box, nlf_boxes = ms_gpu(lambda: build_boxes_list_from_ultralytics(yolo_results, device=device))

            # 6) NLF batched GPU timing
            # def nlf_call():
            #     return nlf.estimate_poses_batched(
            #         img_tensor, nlf_boxes,
            #         intrinsic_matrix=intrinsics,
            #         extrinsic_matrix=extrinsics,
            #         world_up_vector=world_up,
            #         weights=weights,
            #         num_aug=1,
            #     )

            # t_nlf, outputs = ms_gpu(nlf_call)
            t_nlf = 0

            # Aggregate (approx)
            t_sum = t_cpu_pre + t_h2d + t_yolo + t_box + t_nlf

            if read_steps >= args.warmup:
                T["cpu_preproc_ms"].append(t_cpu_pre)
                T["gpu_h2d_norm_ms"].append(t_h2d)
                T["gpu_yolo_ms"].append(t_yolo)
                T["gpu_box_ms"].append(t_box)
                T["gpu_nlf_ms"].append(t_nlf)
                T["sum_components_ms"].append(t_sum)

            read_steps += 1
            if read_steps in [args.warmup, args.warmup + 50, args.warmup + 100, args.warmup + args.iters]:
                done = max(0, read_steps - args.warmup)
                print(f"progress: {done}/{args.iters}")

    for cap in caps:
        cap.release()

    print("\n=== RESULTS (measured steps only; one step = one synchronized multi-cam timestep) ===")
    summarize("CPU preproc (all cams: cvtColor+tensorize)", T["cpu_preproc_ms"])
    summarize("GPU H2D+normalize (batched)", T["gpu_h2d_norm_ms"])
    summarize("GPU YOLO predict (batched)", T["gpu_yolo_ms"])
    summarize("GPU box formatting (batched)", T["gpu_box_ms"])
    summarize("GPU NLF estimate (batched)", T["gpu_nlf_ms"])
    summarize("Sum of components (approx total timestep)", T["sum_components_ms"])

    print("\nNotes:")
    print("- If YOLO/NLF times scale poorly with cams, you may be memory-bound or engine is not truly batched.")
    print("- For maximum throughput, keep drawing/encoding out of the hot loop.\n")


if __name__ == "__main__":
    main()
