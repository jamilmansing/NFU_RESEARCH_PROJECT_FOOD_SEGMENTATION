from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.inference import ADE20K_CHECKPOINT, run_ade20k_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer ADE20K inference on one image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument(
        "--output",
        default="outputs/inference/ade20k_overlay.png",
        help="Path where the overlay image will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        default=ADE20K_CHECKPOINT,
        help="Hugging Face checkpoint to use for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_ade20k_inference(
        image_path=args.image,
        output_path=args.output,
        checkpoint=args.checkpoint,
    )
    print(f"Saved SegFormer overlay to: {output_path}")


if __name__ == "__main__":
    main()

