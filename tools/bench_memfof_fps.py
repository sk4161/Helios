"""
Benchmark MEMFOF (RAFT-style multi-frame optical flow) throughput on a video.

Replicates the sliding 3-frame window + fmap_cache inference from memfof/demo.py,
but times ONLY the model forward passes (video decode and flow-viz/ffmpeg excluded),
with a warmup pass before the measured pass. Prints frames-per-second.

Usage:
  python tools/bench_memfof_fps.py --video <mp4> [--model MEMFOF-Tartan-T-TSKH]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "memfof"))
from memfof.model import MEMFOF, AVAILABLE_MODELS  # noqa: E402


def load_frames(path: str, device: torch.device):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        frames.append(t.to(device))
    cap.release()
    h, w = frames[0].shape[-2:]
    return frames, h, w


@torch.inference_mode()
def run_pass(model, frames):
    """One full sliding-window pass; returns number of model() calls (flow frames)."""
    fmap_cache = [None] * 3
    window = [frames[0]]  # demo duplicates the first frame
    n_calls = 0
    for f in frames:
        window.append(f)
        if len(window) != 3:
            continue
        frames_tensor = torch.stack(window, dim=1)  # [1, 3, C, H, W]
        output = model(frames_tensor, fmap_cache=fmap_cache)
        _ = output["flow"][-1][:, 1]  # forward flow (kept to mirror real use)
        n_calls += 1
        fmap_cache = output["fmap_cache"]
        fmap_cache.pop(0)
        fmap_cache.append(None)
        window.pop(0)
    return n_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="MEMFOF-Tartan-T-TSKH", choices=AVAILABLE_MODELS)
    # Match sam2/sam2/benchmark.py procedure: warm_up runs then average over the rest.
    ap.add_argument("--warm_up", type=int, default=5)
    ap.add_argument("--runs", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    model = MEMFOF.from_pretrained(f"egorchistov/optical-flow-{args.model}").eval().to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model} ({nparams:.1f}M params)")

    frames, h, w = load_frames(args.video, device)
    print(f"Video: {args.video}  ->  {len(frames)} frames @ {w}x{h}")

    # Procedure mirrors sam2/sam2/benchmark.py: run the full-clip pass `runs`
    # times, discard the first `warm_up`, average FPS = count * num_frames / total.
    num_frames = len(frames) - 1  # flow frames produced per run
    total, count, warmup_fps = 0.0, 0, None
    for i in tqdm(range(args.runs), desc=f"Benchmarking {args.model}"):
        start = time.time()
        run_pass(model, frames)
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
        print(f"[MEMFOF-FPS] Warmup FPS: {warmup_fps:.2f} ({args.warm_up} runs)")
    print(f"[MEMFOF-FPS] {num_frames} flow frames, avg over {count} runs -> {fps:.2f} FPS "
          f"({w}x{h}, peak {peak:.2f} GB)")
    print("=" * 50)


if __name__ == "__main__":
    main()
