"""Generate N i2v videos with random seeds and measure pure generation time.

Same model / settings / compile + warmup discipline as tools/progressive_pruning.py, so the
per-video generation time is directly comparable to the progressive-pruning `gen` stage
(load AND compile excluded via an untimed warmup). This is the "just generate 4 directly"
baseline to contrast with "generate 64 then prune to 4".
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_ROOT / ".torchinductor_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from diffusers.models import AutoencoderKLWan  # noqa: E402
from diffusers.utils import load_image  # noqa: E402

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline  # noqa: E402
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler  # noqa: E402
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel  # noqa: E402
from helios.modules.helios_kernels import (  # noqa: E402
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from progressive_pruning import frames_to_uint8, write_mp4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="BestWishYsh/Helios-Distilled")
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num_frames", type=int, default=99)        # 3 chunks -> 97 px frames
    ap.add_argument("--pyramid_steps", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--enable_compile", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--seed_base", type=int, default=-1, help=">=0 for reproducible seeds; <0 = random")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] Helios pipeline...")
    dtype = torch.bfloat16
    transformer = HeliosTransformer3DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception:
        transformer.set_attention_backend("flash_hub")
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(args.model, transformer=transformer, vae=vae,
                                          scheduler=scheduler, torch_dtype=dtype).to(device)
    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    image = load_image(args.image_path).resize((args.width, args.height))

    gen_kwargs = dict(prompt=args.prompt, negative_prompt=None, height=args.height, width=args.width,
                      num_frames=args.num_frames, guidance_scale=1.0, is_enable_stage2=True,
                      pyramid_num_inference_steps_list=args.pyramid_steps, is_amplify_first_chunk=True,
                      num_latent_frames_per_chunk=9, history_sizes=[16, 2, 1], keep_first_frame=True,
                      image_noise_sigma_min=0.111, image_noise_sigma_max=0.135,
                      output_type="np", return_dict=False)

    # ---- untimed warmup: one full generation compiles first-chunk + resume-chunk graphs ----
    if args.enable_compile and args.warmup:
        print("[warmup] compiling via 1 dummy full generation (untimed)...")
        t_w = time.perf_counter()
        wg = torch.Generator(device=device).manual_seed(10**6)
        _ = pipe(image=image, generator=wg, **gen_kwargs)
        torch.cuda.synchronize()
        print(f"[warmup] done in {time.perf_counter() - t_w:.1f}s (excluded)")

    # ---- choose seeds ----
    if args.seed_base >= 0:
        seeds = [args.seed_base + i for i in range(args.n)]
    else:
        seeds = random.sample(range(1, 10**6), args.n)
    print(f"seeds = {seeds}")

    # ---- timed generation (load/compile excluded) ----
    per_seed = []
    torch.cuda.synchronize(); t_all = time.perf_counter()
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(s)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        video_np, = pipe(image=image, generator=g, **gen_kwargs)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        frames = frames_to_uint8(video_np)
        write_mp4(frames, str(out_dir / f"seed{s}.mp4"))
        per_seed.append({"seed": s, "gen_s": round(dt, 2), "frames": int(frames.shape[0])})
        print(f"[gen] seed{s}: {dt:.2f}s  ({frames.shape[0]} frames)")
    torch.cuda.synchronize(); total = time.perf_counter() - t_all

    result = {"n": args.n, "seeds": seeds, "num_frames": args.num_frames,
              "per_seed": per_seed, "total_gen_s": round(total, 2),
              "mean_per_video_s": round(total / args.n, 2)}
    (out_dir / "gen_timing.json").write_text(json.dumps(result, indent=2))
    print(f"\n[TOTAL] {args.n} videos generated in {total:.2f}s "
          f"(mean {total/args.n:.2f}s/video, load+compile excluded)")


if __name__ == "__main__":
    main()
