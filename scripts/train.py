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
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=6e-5, help="Trainer learning rate.")
    parser.add_argument("--image-size", type=int, default=512, help="Square image/mask size.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Print progress every N steps.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Batches prefetched per DataLoader worker. Only used when num workers > 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs. Uses more RAM.",
    )
    parser.add_argument("--test-split", default="test", help="Held-out split to evaluate after training.")
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Do not run held-out test evaluation after training.",
    )
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
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        weight_decay=args.weight_decay,
        resume_from_checkpoint=args.resume_from_checkpoint,
        logging_steps=args.logging_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        run_test_eval=not args.skip_test_eval,
        test_split=args.test_split,
    )
    print(f"Saved final Hugging Face model to: {final_dir}")


if __name__ == "__main__":
    main()
