from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: Path
    mask_path: Path
    split: str


def load_manifest(manifest_path: str | Path, split: str) -> list[SegmentationRecord]:
    """Load image/mask rows from a CSV manifest."""
    path = Path(manifest_path)
    base_dir = path.parent
    records: list[SegmentationRecord] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"image_path", "mask_path", "split"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        for row in reader:
            if row["split"] != split:
                continue

            image_path = _resolve_path(base_dir, row["image_path"])
            mask_path = _resolve_path(base_dir, row["mask_path"])
            records.append(
                SegmentationRecord(
                    image_path=image_path,
                    mask_path=mask_path,
                    split=row["split"],
                )
            )

    if not records:
        raise ValueError(f"No records found for split '{split}' in {path}")

    return records


class SegmentationCsvDataset(Dataset):
    """CSV-backed semantic segmentation dataset for SegFormer."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        processor,
        valid_label_ids: set[int],
        color_to_label_id: dict[tuple[int, int, int], int] | None = None,
    ) -> None:
        self.records = load_manifest(manifest_path, split)
        self.processor = processor
        self.valid_label_ids = valid_label_ids
        self.color_to_label_id = color_to_label_id or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path)
        mask_array = self._load_mask_array(mask, record.mask_path)

        unknown_ids = set(np.unique(mask_array).tolist()) - self.valid_label_ids
        if unknown_ids:
            raise ValueError(
                f"Mask {record.mask_path} contains label ids not in labels.json: "
                f"{sorted(unknown_ids)}"
            )

        encoded = self.processor(
            images=image,
            segmentation_maps=mask_array,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
        }

    def _load_mask_array(self, mask: Image.Image, mask_path: Path) -> np.ndarray:
        if mask.mode in {"RGB", "RGBA", "P"}:
            if self.color_to_label_id:
                return rgb_mask_to_label_ids(mask.convert("RGB"), self.color_to_label_id, mask_path)

            if mask.mode == "P":
                mask_array = np.array(mask, dtype=np.int64)
            else:
                raise ValueError(
                    f"Mask {mask_path} is RGB/RGBA but labels.json does not include colors. "
                    "Add color entries or convert masks to class-id grayscale."
                )
        else:
            mask_array = np.array(mask.convert("L"), dtype=np.int64)

        unknown_ids = set(np.unique(mask_array).tolist()) - self.valid_label_ids
        if unknown_ids:
            raise ValueError(
                f"Mask {mask_path} contains label ids not in labels.json: {sorted(unknown_ids)}"
            )
        return mask_array


def segmentation_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def rgb_mask_to_label_ids(
    mask: Image.Image,
    color_to_label_id: dict[tuple[int, int, int], int],
    mask_path: Path,
) -> np.ndarray:
    mask_rgb = np.array(mask, dtype=np.uint8)
    label_mask = np.full(mask_rgb.shape[:2], fill_value=-1, dtype=np.int64)

    for rgb, label_id in color_to_label_id.items():
        color_matches = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        label_mask[color_matches] = label_id

    if np.any(label_mask == -1):
        unknown_colors = np.unique(mask_rgb[label_mask == -1].reshape(-1, 3), axis=0)
        preview = [tuple(int(channel) for channel in color) for color in unknown_colors[:10]]
        raise ValueError(
            f"Mask {mask_path} contains RGB colors not in labels.json: {preview}"
        )

    return label_mask


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path
