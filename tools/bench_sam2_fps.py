"""
Benchmark native SAM2 video-tracking throughput on a single video.

Mirrors tools/track_videos_with_sam2_native.py's optimized setup
(bf16 autocast + TF32 + vos_optimized torch.compile, mode=default) but seeds
with a fixed bbox (no GroundingDINO) and times the propagate_in_video loop with
a warmup pass first. Prints frames-per-second of the tracker.

Usage:
  python tools/bench_sam2_fps.py --video <mp4> [--bbox X1 Y1 X2 Y2] [--no_compile]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HELIOS_ROOT = Path(__file__).resolve().parent.parent

# Work around sam2's parent-dir shadow check: the installed `sam2` package must
# win over the ./sam2 submodule. Do this before importing sam2.
import os  # noqa: E402
os.chdir("/tmp")
if sys.path and (sys.path[0] == "" or Path(sys.path[0]).resolve() == _HELIOS_ROOT):
    sys.path = sys.path[1:]


def patch_vos_compile_mode():
    """mode='default' (torch 2.10 max-autotune triggers a CUDAGraph error on
    SAM2's memory_attention)."""
    import sam2.sam2_video_predictor as _svp

    def _patched(self):
        print("[bench_sam2] vos_optimized compile (mode=default)...")
        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward, mode="default", fullgraph=True, dynamic=False)
        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward, mode="default", fullgraph=True, dynamic=True)
        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward, mode="default", fullgraph=True, dynamic=False)
        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward, mode="default", fullgraph=True, dynamic=False)

    _svp.SAM2VideoPredictorVOS._compile_all_components = _patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, default=[200, 110, 470, 300],
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Seed box in video pixel coords (default ~centered for the fish).")
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--sam2_ckpt",
                    default=str(_HELIOS_ROOT / "sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    ap.add_argument("--no_compile", action="store_true")
    # Match sam2/sam2/benchmark.py procedure: warm_up runs then average over the rest.
    ap.add_argument("--warm_up", type=int, default=5)
    ap.add_argument("--runs", type=int, default=25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.no_compile:
        patch_vos_compile_mode()

    from sam2.build_sam import build_sam2_video_predictor
    ckpt = args.sam2_ckpt
    if not Path(ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    print(f"Building SAM2 (cfg={args.sam2_cfg}, vos_optimized={not args.no_compile})...")
    predictor = build_sam2_video_predictor(
        args.sam2_cfg, ckpt, device=device, vos_optimized=not args.no_compile)

    vpath = args.video
    if not Path(vpath).is_absolute():
        vpath = str(_HELIOS_ROOT / vpath)
    print(f"Video: {vpath}  bbox={args.bbox}")

    # Benchmark procedure mirrors sam2/sam2/benchmark.py:
    #   init_state + seed ONCE, then run propagate_in_video over the full clip
    #   `runs` times; the first `warm_up` runs are discarded, FPS is averaged
    #   over the remaining runs as count * num_frames / total_time.
    inference_state = predictor.init_state(video_path=vpath)
    num_frames = inference_state["num_frames"]
    predictor.add_new_points_or_box(
        inference_state=inference_state, frame_idx=0, obj_id=1,
        box=np.array(args.bbox, dtype=np.float32),
    )

    total, count, warmup_fps = 0.0, 0, None
    with torch.inference_mode():
        for i in tqdm(range(args.runs), desc="Benchmarking SAM2"):
            start = time.time()
            for _out in predictor.propagate_in_video(inference_state):
                pass
            total += time.time() - start
            count += 1
            if i == args.warm_up - 1:
                warmup_fps = count * num_frames / total
                total, count = 0.0, 0
    fps = count * num_frames / total
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else float("nan")
    print("=" * 50)
    if warmup_fps is not None:
        print(f"[SAM2-FPS] Warmup FPS: {warmup_fps:.2f} ({args.warm_up} runs)")
    print(f"[SAM2-FPS] {num_frames} frames, avg over {count} runs -> {fps:.2f} FPS "
          f"(vos_optimized={not args.no_compile}, peak {peak:.2f} GB)")
    print("=" * 50)


if __name__ == "__main__":
    main()
