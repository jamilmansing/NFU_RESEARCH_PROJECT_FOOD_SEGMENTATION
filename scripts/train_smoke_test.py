from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.training import MIT_B0_CHECKPOINT, run_training_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test SegFormer MIT-B0 training.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path, mask_path, split.")
    parser.add_argument(
        "--labels",
        default="configs/labels.json",
        help="JSON mapping class ids to class names.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/segformer-mit-b0-food-smoke",
        help="Directory for Trainer outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=MIT_B0_CHECKPOINT,
        help="Hugging Face checkpoint to fine-tune.",
    )
    parser.add_argument("--max-steps", type=int, default=5, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training_smoke_test(
        manifest_path=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
