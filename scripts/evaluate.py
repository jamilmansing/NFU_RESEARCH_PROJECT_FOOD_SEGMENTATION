from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.evaluation import plot_training_history, run_test_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SegFormer model on a held-out split.")
    parser.add_argument("--model-dir", required=True, help="Saved Hugging Face model directory.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path, mask_path, split.")
    parser.add_argument("--labels", default="configs/labels.json", help="Label JSON used for training.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and plots.")
    parser.add_argument("--split", default="test", help="Manifest split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Override processor image size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Prefetch factor when workers > 0.")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive.")
    parser.add_argument(
        "--training-run-dir",
        default=None,
        help="Optional Trainer output directory containing trainer_state.json for training plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_test_evaluation(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    if args.training_run_dir:
        plot_training_history(args.training_run_dir, Path(args.output_dir) / "plots")


if __name__ == "__main__":
    main()
