"""
Shared helpers for the SAM2 tools (mask_with_sam2.py, track_videos_with_sam2_native.py):
GroundingDINO text->box detection and the sam2/sam2/benchmark.py compute settings
(bf16 autocast + TF32). Kept dependency-light: transformers is imported lazily so
importing this module never pulls in heavy deps or conflicts with the native
sam2 package's parent-dir shadow check.
"""
import numpy as np
import torch


def enable_tf32(device):
    """Enable TF32 on Ampere+ GPUs, as in sam2/sam2/benchmark.py."""
    if (str(device).startswith("cuda") and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 8):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def detect_boxes_with_gdino(image, text, gdino_model, device,
                            box_thr=0.35, text_thr=0.25, use_amp=True):
    """Run GroundingDINO once. Returns (boxes [N,4] np, scores [N] np, labels list).
    Raises if nothing is detected. Forward runs under bf16 autocast on CUDA."""
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    processor = GroundingDinoProcessor.from_pretrained(gdino_model)
    model = GroundingDinoForObjectDetection.from_pretrained(gdino_model).to(device)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    amp = str(device).startswith("cuda") and use_amp
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = res["boxes"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    labels = res["text_labels"] if "text_labels" in res else res.get("labels", [])
    del model, processor
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if len(boxes) == 0:
        raise RuntimeError(f"GroundingDINO found no boxes for prompt {text!r} "
                           "(try lowering --box_threshold or rephrasing)")
    return boxes, scores, labels


def best_box(boxes, scores):
    """Return the highest-scoring box as list[4] and print it."""
    i = int(np.argmax(scores))
    bbox = boxes[i].tolist()
    print(f"GroundingDINO bbox: {bbox} (score={scores[i]:.3f})")
    return bbox
