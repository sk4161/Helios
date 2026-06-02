"""
Benchmark canonical RAFT (torchvision) optical-flow throughput on a video.

This is the original RAFT (Teed & Deng, 2020) as shipped in torchvision, distinct
from the RAFT-style multi-frame MEMFOF in ./memfof. Processes consecutive frame
pairs (img_t, img_t+1), times only the model forward (12 flow-update iters,
torchvision default) with a warmup pass first. Prints frames-per-second.

Usage:
  python tools/bench_raft_fps.py --video <mp4> [--model large|small]
"""
import argparse
import time

import cv2
import torch
from tqdm import tqdm
from torchvision.models.optical_flow import (
    raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights,
)


def load_frames_uint8(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)  # [C,H,W] uint8
        frames.append(t)
    cap.release()
    return frames


@torch.inference_mode()
def run_pass(model, pairs):
    n = 0
    for img1, img2 in pairs:
        flows = model(img1, img2)  # list of iterative updates
        _ = flows[-1]
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="large", choices=["large", "small"])
    # Match sam2/sam2/benchmark.py procedure: warm_up runs then average over the rest.
    ap.add_argument("--warm_up", type=int, default=5)
    ap.add_argument("--runs", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # TF32 on Ampere+ (same as the SAM2 benchmark procedure).
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if args.model == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)
    model = model.eval().to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: RAFT-{args.model} ({nparams:.1f}M params, weights={weights})")

    tfm = weights.transforms()
    frames = load_frames_uint8(args.video)
    h, w = frames[0].shape[-2:]
    print(f"Video: {args.video}  ->  {len(frames)} frames @ {w}x{h}")

    # Pre-build normalized consecutive pairs on GPU (decode/transform excluded from timing)
    pairs = []
    for i in range(len(frames) - 1):
        a = frames[i].unsqueeze(0)
        b = frames[i + 1].unsqueeze(0)
        a, b = tfm(a, b)
        pairs.append((a.to(device), b.to(device)))

    # Procedure mirrors sam2/sam2/benchmark.py: run the full-clip pass `runs`
    # times, discard the first `warm_up`, average FPS = count * num_frames / total.
    num_frames = len(pairs)  # flow frames produced per run
    total, count, warmup_fps = 0.0, 0, None
    for i in tqdm(range(args.runs), desc=f"Benchmarking RAFT-{args.model}"):
        start = time.time()
        run_pass(model, pairs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total += time.time() - start
        count += 1
        if i == args.warm_up - 1:
            warmup_fps = count * num_frames / total
            total, count = 0.0, 0

    fps = count * num_frames / total
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else float("nan")
    print("=" * 50)
    if warmup_fps is not None:
        print(f"[RAFT-FPS] RAFT-{args.model} Warmup FPS: {warmup_fps:.2f} ({args.warm_up} runs)")
    print(f"[RAFT-FPS] RAFT-{args.model}: {num_frames} flow frames, avg over {count} runs "
          f"-> {fps:.2f} FPS ({w}x{h}, peak {peak:.2f} GB)")
    print("=" * 50)


if __name__ == "__main__":
    main()
