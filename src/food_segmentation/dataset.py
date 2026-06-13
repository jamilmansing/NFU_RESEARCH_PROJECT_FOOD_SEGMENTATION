from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from random import random

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
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path.resolve()}")

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


def validate_manifest(
    manifest_path: str | Path,
    splits: tuple[str, ...],
    valid_label_ids: set[int],
    color_to_label_id: dict[tuple[int, int, int], int] | None = None,
    progress_interval: int = 100,
) -> None:
    """Validate dataset paths and mask labels before model loading."""
    color_to_label_id = color_to_label_id or {}
    for split in splits:
        records = load_manifest(manifest_path, split)
        print(f"Validating {len(records)} {split} records from {manifest_path}...", flush=True)
        for index, record in enumerate(records, start=1):
            if not record.image_path.exists():
                raise FileNotFoundError(f"Image file not found: {record.image_path.resolve()}")
            if not record.mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {record.mask_path.resolve()}")

            with Image.open(record.mask_path) as mask:
                validate_mask_labels(
                    mask=mask,
                    mask_path=record.mask_path,
                    valid_label_ids=valid_label_ids,
                    color_to_label_id=color_to_label_id,
                )
            if progress_interval > 0 and index % progress_interval == 0:
                print(f"Validated {index}/{len(records)} {split} records...", flush=True)
        print(f"Finished validating {split} split.", flush=True)


class SegmentationCsvDataset(Dataset):
    """CSV-backed semantic segmentation dataset for SegFormer."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        processor,
        valid_label_ids: set[int],
        color_to_label_id: dict[tuple[int, int, int], int] | None = None,
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.records = load_manifest(manifest_path, split)
        self.processor = processor
        self.valid_label_ids = valid_label_ids
        self.color_to_label_id = color_to_label_id or {}
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path)
        mask_array = self._load_mask_array(mask, record.mask_path)

        if self.augment and random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask_array = np.fliplr(mask_array).copy()

        unknown_ids = set(np.unique(mask_array).tolist()) - self.valid_label_ids
        if unknown_ids:
            raise ValueError(
                f"Mask {record.mask_path} contains label ids not in labels.json: "
                f"{sorted(unknown_ids)}"
            )

        encoded = self.processor(
            images=image,
            segmentation_maps=mask_array,
            size={"height": self.image_size, "width": self.image_size},
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
    packed_mask = pack_rgb(mask_rgb)
    unique_colors, inverse = np.unique(packed_mask.reshape(-1), return_inverse=True)

    packed_color_to_label_id = {
        pack_rgb_tuple(rgb): label_id
        for rgb, label_id in color_to_label_id.items()
    }
    unique_label_ids = np.full(unique_colors.shape, fill_value=-1, dtype=np.int64)
    for index, packed_color in enumerate(unique_colors):
        unique_label_ids[index] = packed_color_to_label_id.get(int(packed_color), -1)

    unknown_packed_colors = unique_colors[unique_label_ids == -1]
    if unknown_packed_colors.size:
        preview = [unpack_rgb(int(color)) for color in unknown_packed_colors[:10]]
        raise ValueError(
            f"Mask {mask_path} contains RGB colors not in labels.json: {preview}"
        )

    return unique_label_ids[inverse].reshape(mask_rgb.shape[:2])


def pack_rgb(mask_rgb: np.ndarray) -> np.ndarray:
    rgb = mask_rgb.astype(np.uint32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def pack_rgb_tuple(rgb: tuple[int, int, int]) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def unpack_rgb(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def validate_mask_labels(
    mask: Image.Image,
    mask_path: Path,
    valid_label_ids: set[int],
    color_to_label_id: dict[tuple[int, int, int], int],
) -> None:
    if mask.mode in {"RGB", "RGBA"}:
        if not color_to_label_id:
            raise ValueError(
                f"Mask {mask_path} is RGB/RGBA but labels.json does not include colors."
            )
        rgb_mask_to_label_ids(mask.convert("RGB"), color_to_label_id, mask_path)
        return

    if mask.mode == "P" and color_to_label_id:
        rgb_mask_to_label_ids(mask.convert("RGB"), color_to_label_id, mask_path)
        return

    mask_array = np.array(mask.convert("L"), dtype=np.int64)
    unknown_ids = set(np.unique(mask_array).tolist()) - valid_label_ids
    if unknown_ids:
        raise ValueError(f"Mask {mask_path} contains label ids not in labels.json: {sorted(unknown_ids)}")


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path
