from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.training import DEFAULT_OUTPUT_DIR, MIT_B0_CHECKPOINT, run_full_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer MIT-B0 on a custom dataset.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path, mask_path, split.")
    parser.add_argument(
        "--labels",
        default="configs/labels.json",
        help="JSON mapping class ids to names and optional RGB colors.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and the final Hugging Face model.",
    )
    parser.add_argument(
        "--checkpoint",
        default=MIT_B0_CHECKPOINT,
        help="Base checkpoint or resumeable model directory. Defaults to nvidia/mit-b0.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--learning-rate", type=float, default=6e-5, help="Trainer learning rate.")
    parser.add_argument("--image-size", type=int, default=512, help="Square image/mask size.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a Trainer checkpoint directory to continue training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir = run_full_training(
        manifest_path=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        weight_decay=args.weight_decay,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Saved final Hugging Face model to: {final_dir}")


if __name__ == "__main__":
    main()
