"""
Detect an object in an image with GroundingDINO and segment it with SAM2.

Outputs:
  - <stem>_mask.png       : binary mask (white = object)
  - <stem>_masked.png     : input with non-object region blacked out
  - <stem>_overlay.png    : input with semi-transparent colored mask + bbox (verification)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam2Model, Sam2Processor

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make sam2_common importable
from sam2_common import detect_boxes_with_gdino, enable_tf32  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--text", type=str, required=True,
                   help="Detection prompt for GroundingDINO. Use lowercase, end with '.', e.g. 'a cat.'")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same as input image)")
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2_model", type=str, default="facebook/sam2-hiera-base-plus")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Match sam2/sam2/benchmark.py compute settings: bf16 autocast + TF32 on Ampere+.
    use_amp = args.device.startswith("cuda")
    enable_tf32(args.device)

    image_path = Path(args.image_path)
    out_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image.size}")

    # ---------- 1. GroundingDINO: text -> boxes ----------
    print(f"Detecting '{args.text}' with GroundingDINO ({args.gdino_model})...")
    boxes, scores, labels = detect_boxes_with_gdino(
        image, args.text, args.gdino_model, args.device,
        args.box_threshold, args.text_threshold, use_amp=use_amp,
    )
    print(f"Found {len(boxes)} boxes:")
    for b, s, l in zip(boxes, scores, labels):
        print(f"  {l} score={s:.3f} box={b.tolist()}")

    # ---------- 2. SAM2: bbox -> mask ----------
    print(f"Loading SAM2 ({args.sam2_model})...")
    sam_processor = Sam2Processor.from_pretrained(args.sam2_model)
    sam_model = Sam2Model.from_pretrained(args.sam2_model).to(args.device)

    # SAM2 expects boxes per image as a list of [x1, y1, x2, y2]
    sam_inputs = sam_processor(
        images=image,
        input_boxes=[boxes.tolist()],
        return_tensors="pt",
    ).to(args.device)

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
        sam_outputs = sam_model(**sam_inputs, multimask_output=False)

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        original_sizes=sam_inputs["original_sizes"],
    )[0]  # tensor [num_boxes, 1, H, W] or list

    # Normalize mask shape -> numpy [H, W] uint8 (union of all detected boxes)
    if isinstance(masks, torch.Tensor):
        masks_np = masks.numpy()
    else:
        masks_np = np.stack([m.numpy() if isinstance(m, torch.Tensor) else m for m in masks])
    # collapse channel dim if any
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]  # [N, H, W]
    union_mask = (masks_np > 0).any(axis=0).astype(np.uint8) * 255

    # ---------- 3. Save outputs ----------
    mask_img = Image.fromarray(union_mask, mode="L")
    mask_path = out_dir / f"{stem}_mask.png"
    mask_img.save(mask_path)
    print(f"Saved mask: {mask_path}")

    # masked image (black background)
    image_np = np.array(image)
    masked_np = image_np.copy()
    masked_np[union_mask == 0] = 0
    Image.fromarray(masked_np).save(out_dir / f"{stem}_masked.png")
    print(f"Saved masked: {out_dir / f'{stem}_masked.png'}")

    # overlay visualization (semi-transparent red mask + bboxes)
    overlay = image.convert("RGBA")
    color_layer = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
    color_arr = np.zeros((*union_mask.shape, 4), dtype=np.uint8)
    color_arr[union_mask > 0] = [255, 0, 0, 110]  # red, ~43% alpha
    color_layer = Image.fromarray(color_arr, mode="RGBA")
    overlay = Image.alpha_composite(overlay, color_layer)
    draw = ImageDraw.Draw(overlay)
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
        draw.text((x1 + 4, y1 + 4), f"{l} {s:.2f}", fill=(0, 255, 0, 255))
    overlay_path = out_dir / f"{stem}_overlay.png"
    overlay.convert("RGB").save(overlay_path)
    print(f"Saved overlay: {overlay_path}")


if __name__ == "__main__":
    main()
