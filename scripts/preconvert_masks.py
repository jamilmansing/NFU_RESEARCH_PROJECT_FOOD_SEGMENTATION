from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_segmentation.preconvert import preconvert_rgb_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preconvert RGB segmentation masks to class-ID masks.")
    parser.add_argument("--manifest", required=True, help="Input CSV with image_path, mask_path, split.")
    parser.add_argument(
        "--labels",
        default="configs/labels.json",
        help="JSON mapping class ids to names and RGB colors.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where converted masks will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="CSV path to write with mask_path pointing at converted masks.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "png8", "png16", "npy"],
        default="auto",
        help="Converted mask format. auto uses png8 up to 256 labels, otherwise png16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_manifest = preconvert_rgb_masks(
        manifest_path=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        output_format=args.format,
    )
    print(f"Saved converted manifest to: {output_manifest}")


if __name__ == "__main__":
    main()
