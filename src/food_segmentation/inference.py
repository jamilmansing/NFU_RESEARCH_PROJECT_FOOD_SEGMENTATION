from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


ADE20K_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"


def run_ade20k_inference(
    image_path: str | Path,
    output_path: str | Path,
    checkpoint: str = ADE20K_CHECKPOINT,
    alpha: float = 0.55,
) -> Path:
    """Run the ADE20K-finetuned SegFormer checkpoint and save an overlay image."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    overlay = make_overlay(image, predicted_mask, alpha=alpha)
    overlay.save(output_path)
    return output_path


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    image_array = np.array(image).astype(np.float32)
    color_mask = palette_for_mask(mask).astype(np.float32)
    blended = ((1.0 - alpha) * image_array + alpha * color_mask).clip(0, 255)
    return Image.fromarray(blended.astype(np.uint8))


def palette_for_mask(mask: np.ndarray) -> np.ndarray:
    """Create a stable color palette without depending on ADE20K metadata."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for label_id in range(256):
        palette[label_id] = [
            (37 * label_id) % 255,
            (17 * label_id + 97) % 255,
            (29 * label_id + 53) % 255,
        ]
    return palette[mask]

