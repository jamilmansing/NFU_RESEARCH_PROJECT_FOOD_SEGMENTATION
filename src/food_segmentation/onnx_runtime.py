from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForSemanticSegmentation

from food_segmentation.config import load_label_maps
from food_segmentation.inference import ADE20K_CHECKPOINT, make_overlay
from food_segmentation.training import MIT_B0_CHECKPOINT


class SegFormerOnnxWrapper(nn.Module):
    """Return logits directly so ONNX Runtime gets a plain tensor output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export_segformer_to_onnx(
    checkpoint: str,
    output_path: str | Path,
    labels_path: str | Path | None = None,
    image_size: int = 512,
    opset: int = 17,
) -> Path:
    """Export a SegFormer semantic segmentation model to ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_kwargs = {}
    if labels_path is not None:
        id2label, label2id = load_label_maps(labels_path)
        config = AutoConfig.from_pretrained(checkpoint)
        config.num_labels = len(id2label)
        config.id2label = id2label
        config.label2id = label2id
        model_kwargs = {
            "config": config,
            "ignore_mismatched_sizes": True,
        }

    model = AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint,
        **model_kwargs,
    )
    model.eval()
    wrapped_model = SegFormerOnnxWrapper(model)
    wrapped_model.eval()
    dummy_pixel_values = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        wrapped_model,
        (dummy_pixel_values,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "logit_height", 3: "logit_width"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    return output_path


def run_onnx_segformer_inference(
    onnx_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    processor_checkpoint: str = ADE20K_CHECKPOINT,
    use_cuda: bool = False,
    alpha: float = 0.55,
) -> Path:
    """Run a SegFormer ONNX model and save a segmentation overlay."""
    onnx_path = Path(onnx_path)
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    providers = _execution_providers(use_cuda)
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    processor = AutoImageProcessor.from_pretrained(processor_checkpoint)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="np")
    logits = session.run(["logits"], {"pixel_values": inputs["pixel_values"]})[0]

    logits_tensor = torch.from_numpy(logits)
    upsampled_logits = F.interpolate(
        logits_tensor,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    predicted_mask = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)

    overlay = make_overlay(image, predicted_mask, alpha=alpha)
    overlay.save(output_path)
    return output_path


def _execution_providers(use_cuda: bool) -> list[str]:
    available_providers = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def default_export_checkpoint(kind: str) -> str:
    if kind == "ade20k":
        return ADE20K_CHECKPOINT
    if kind == "mit-b0":
        return MIT_B0_CHECKPOINT
    raise ValueError(f"Unknown export kind: {kind}")
