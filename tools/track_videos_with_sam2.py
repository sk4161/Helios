"""
Track an object across all frames of every mp4 in --video_dir using SAM2 video.

Bootstrap each video with the same first-frame bounding box from --bbox or by
running GroundingDINO once on --reference_image with --text. The resulting
per-frame masks are saved as bool tensors `[T, H, W]` to --output_dir/seed{N}.pt
for use as time-varying masks by tools/select_diverse_videos.py.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from video_reader import PyVideoReader

from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    Sam2VideoModel,
    Sam2VideoProcessor,
)

SEED_RE = re.compile(r"seed(\d+)_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--reference_image", type=str, default=None,
                   help="Image to detect the object on (e.g. the first frame). If not given, the first frame of the first video is used.")
    p.add_argument("--text", type=str, default="a cat.",
                   help="GroundingDINO text prompt for detecting the object.")
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Skip GroundingDINO and use this bbox directly (in video pixel coords).")
    p.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2_video_model", type=str, default="facebook/sam2-hiera-base-plus")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--video_glob", type=str, default="seed*.mp4")
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--text_threshold", type=float, default=0.25)
    return p.parse_args()


def find_videos(video_dir: Path, glob_pat: str):
    out = []
    for p in sorted(video_dir.glob(glob_pat)):
        m = SEED_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_video_frames_pil(path: Path) -> list[Image.Image]:
    """Decode mp4 -> list of PIL Images (RGB)."""
    vr = PyVideoReader(str(path), threads=0)
    buf = vr.decode()  # [T, H, W, 3] uint8
    del vr
    return [Image.fromarray(buf[i]) for i in range(buf.shape[0])]


def detect_bbox_with_gdino(image: Image.Image, text: str, gdino_model: str,
                           device: str, box_thr: float, text_thr: float) -> list[float]:
    """Run GroundingDINO once and return the highest-score bbox."""
    processor = GroundingDinoProcessor.from_pretrained(gdino_model)
    model = GroundingDinoForObjectDetection.from_pretrained(gdino_model).to(device)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO found no boxes for prompt {text!r}")
    best = int(np.argmax(scores))
    bbox = boxes[best].tolist()
    print(f"GroundingDINO bbox: {bbox} (score={scores[best]:.3f})")
    # free
    del model, processor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return bbox


@torch.no_grad()
def track_video(model: Sam2VideoModel, processor: Sam2VideoProcessor,
                frames: list[Image.Image], bbox_xyxy: list[float], device: str) -> torch.Tensor:
    """Run SAM2 video tracking with a single bbox seed at frame 0.
    Returns bool tensor [T, H, W] (True = foreground)."""
    H, W = frames[0].height, frames[0].width
    T = len(frames)

    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        dtype=torch.float32,
    )
    # Add the bbox at frame 0 for object id 1
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=1,
        input_boxes=[[bbox_xyxy]],  # [image][box][4]
        original_size=(H, W),
    )

    # Pre-allocate output as bool to keep memory small
    masks = torch.zeros((T, H, W), dtype=torch.bool)

    # Forward propagation from frame 0
    for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
        frame_idx = int(out.frame_idx)
        # pred_masks: [batch=1, num_objs=1, H_lowres, W_lowres] at model resolution
        pred = out.pred_masks
        # post_process_masks expects a *list* of [N, C, H, W] tensors and a list of (H, W)
        upsampled = processor.post_process_masks(
            [pred],
            original_sizes=[(H, W)],
            binarize=True,
        )[0]  # [1, 1, H, W] bool
        masks[frame_idx] = upsampled[0, 0].to(torch.bool).cpu()

    return masks


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(Path(args.video_dir), args.video_glob)
    if not videos:
        raise SystemExit(f"No videos in {args.video_dir}")
    print(f"Found {len(videos)} videos")

    # Resolve the seed bbox once (shared across all videos since they share frame 0)
    if args.bbox is not None:
        bbox = list(args.bbox)
        print(f"Using user-provided bbox: {bbox}")
    else:
        if args.reference_image:
            ref_img = Image.open(args.reference_image).convert("RGB")
        else:
            ref_img = load_video_frames_pil(videos[0][1])[0]
        # Note: GDino bbox is in the *reference image*'s pixel space.
        # If you supplied --reference_image at a different resolution than the
        # video, that's a problem. We resize the reference to the video size first.
        sample_frames = load_video_frames_pil(videos[0][1])
        H, W = sample_frames[0].height, sample_frames[0].width
        if (ref_img.width, ref_img.height) != (W, H):
            print(f"Resizing reference {ref_img.size} -> ({W}, {H}) to match video")
            ref_img = ref_img.resize((W, H), Image.LANCZOS)
        bbox = detect_bbox_with_gdino(
            ref_img, args.text, args.gdino_model, args.device,
            args.box_threshold, args.text_threshold,
        )

    # Load SAM2 video model once
    print(f"Loading SAM2 video ({args.sam2_video_model})...")
    processor = Sam2VideoProcessor.from_pretrained(args.sam2_video_model)
    model = Sam2VideoModel.from_pretrained(args.sam2_video_model).to(args.device).eval()

    summary = {"bbox": bbox, "videos": {}}
    for seed, path in tqdm(videos, desc="track"):
        out_path = out_dir / f"seed{seed}.pt"
        if out_path.exists():
            mask = torch.load(out_path, map_location="cpu", weights_only=True)
        else:
            frames = load_video_frames_pil(path)
            mask = track_video(model, processor, frames, bbox, args.device)
            torch.save(mask, out_path)
            del frames
        fg_frac = mask.float().mean().item()
        summary["videos"][seed] = {"fg_frac_mean": fg_frac, "T": int(mask.shape[0])}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
