from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.onnx_runtime import default_export_checkpoint, export_segformer_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SegFormer semantic segmentation to ONNX.")
    parser.add_argument(
        "--kind",
        choices=["ade20k", "mit-b0"],
        default="ade20k",
        help="Use ADE20K full model or MIT-B0 encoder with a custom segmentation head.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint or trained model directory. Defaults depend on --kind.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Required for mit-b0/custom exports so the segmentation head knows your classes.",
    )
    parser.add_argument(
        "--output",
        default="outputs/onnx/segformer.onnx",
        help="Where to save the ONNX model.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Dummy export image size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or default_export_checkpoint(args.kind)
    if args.kind == "mit-b0" and args.labels is None:
        raise ValueError("--labels is required when exporting --kind mit-b0.")

    output_path = export_segformer_to_onnx(
        checkpoint=checkpoint,
        output_path=args.output,
        labels_path=args.labels,
        image_size=args.image_size,
        opset=args.opset,
    )
    print(f"Saved ONNX model to: {output_path}")


if __name__ == "__main__":
    main()

