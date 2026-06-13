from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from food_segmentation.config import load_label_colors, load_label_maps
from food_segmentation.dataset import rgb_mask_to_label_ids


def preconvert_rgb_masks(
    manifest_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    output_manifest: str | Path,
    output_format: str = "auto",
) -> Path:
    """Convert RGB semantic masks to class-ID masks and write a new manifest."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_manifest = Path(output_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    id2label, _ = load_label_maps(labels_path)
    color_to_label_id = load_label_colors(labels_path)
    if not color_to_label_id:
        raise ValueError("labels.json must include RGB color entries for mask conversion.")

    mask_format = choose_output_format(output_format, num_labels=len(id2label))
    manifest_rows = read_manifest_rows(manifest_path)
    converted_rows: list[dict[str, str]] = []

    total_rows = len(manifest_rows)
    for index, row in enumerate(manifest_rows, start=1):
        source_mask_path = resolve_manifest_path(manifest_path, row["mask_path"])
        split = row["split"]
        output_mask_path = converted_mask_path(
            output_dir=output_dir,
            split=split,
            source_mask_path=source_mask_path,
            mask_format=mask_format,
        )
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_mask_path) as mask:
            label_mask = rgb_mask_to_label_ids(
                mask=mask.convert("RGB"),
                color_to_label_id=color_to_label_id,
                mask_path=source_mask_path,
            )
        save_label_mask(label_mask, output_mask_path, mask_format)

        converted_row = dict(row)
        converted_row["mask_path"] = relative_manifest_path(output_manifest, output_mask_path)
        converted_rows.append(converted_row)

        if index == 1 or index % 100 == 0 or index == total_rows:
            print(f"Converted {index}/{total_rows} masks...", flush=True)

    write_manifest(output_manifest, converted_rows)
    return output_manifest


def choose_output_format(output_format: str, num_labels: int) -> str:
    if output_format == "auto":
        return "png16" if num_labels > 256 else "png8"
    if output_format not in {"png8", "png16", "npy"}:
        raise ValueError("output_format must be one of: auto, png8, png16, npy")
    if output_format == "png8" and num_labels > 256:
        raise ValueError("png8 supports at most 256 label ids. Use png16, npy, or auto.")
    return output_format


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"image_path", "mask_path", "split"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
        return list(reader)


def write_manifest(output_manifest: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows found in manifest.")

    fieldnames = list(rows[0].keys())
    with output_manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def converted_mask_path(
    output_dir: Path,
    split: str,
    source_mask_path: Path,
    mask_format: str,
) -> Path:
    suffix = ".npy" if mask_format == "npy" else ".png"
    return output_dir / split / f"{source_mask_path.stem}{suffix}"


def save_label_mask(label_mask: np.ndarray, output_path: Path, mask_format: str) -> None:
    if mask_format == "npy":
        np.save(output_path, label_mask.astype(np.int64))
        return

    if mask_format == "png8":
        Image.fromarray(label_mask.astype(np.uint8), mode="L").save(output_path)
        return

    Image.fromarray(label_mask.astype(np.uint16)).save(output_path)


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def relative_manifest_path(output_manifest: Path, target_path: Path) -> str:
    try:
        return target_path.relative_to(output_manifest.parent).as_posix()
    except ValueError:
        return str(target_path)
