from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.inference import ADE20K_CHECKPOINT
from food_segmentation.onnx_runtime import run_onnx_segformer_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer inference with ONNX Runtime.")
    parser.add_argument("--onnx", required=True, help="Path to a SegFormer ONNX model.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument(
        "--output",
        default="outputs/onnx/onnx_overlay.png",
        help="Path where the overlay image will be saved.",
    )
    parser.add_argument(
        "--processor-checkpoint",
        default=ADE20K_CHECKPOINT,
        help="Checkpoint/directory to load image preprocessing settings from.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use ONNX Runtime CUDAExecutionProvider if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_onnx_segformer_inference(
        onnx_path=args.onnx,
        image_path=args.image,
        output_path=args.output,
        processor_checkpoint=args.processor_checkpoint,
        use_cuda=args.cuda,
    )
    print(f"Saved ONNX Runtime overlay to: {output_path}")


if __name__ == "__main__":
    main()
