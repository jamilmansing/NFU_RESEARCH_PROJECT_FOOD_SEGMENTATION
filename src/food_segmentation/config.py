from __future__ import annotations

import json
from pathlib import Path


def load_label_maps(labels_path: str | Path) -> tuple[dict[int, str], dict[str, int]]:
    """Load id2label and label2id maps from a JSON file."""
    path = Path(labels_path)
    with path.open("r", encoding="utf-8") as file:
        raw_labels = json.load(file)

    id2label = {
        int(label_id): _label_name(label_data)
        for label_id, label_data in raw_labels.items()
    }
    label2id = {name: label_id for label_id, name in id2label.items()}

    _validate_contiguous_ids(id2label)
    return id2label, label2id


def load_label_colors(labels_path: str | Path) -> dict[tuple[int, int, int], int]:
    """Load RGB color to label-id mappings when labels.json includes colors."""
    path = Path(labels_path)
    with path.open("r", encoding="utf-8") as file:
        raw_labels = json.load(file)

    color_to_label_id: dict[tuple[int, int, int], int] = {}
    for label_id_text, label_data in raw_labels.items():
        if not isinstance(label_data, dict) or "color" not in label_data:
            continue

        color = label_data["color"]
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError(f"Label {label_id_text} has an invalid RGB color: {color}")

        rgb = tuple(int(channel) for channel in color)
        if rgb in color_to_label_id:
            raise ValueError(f"Duplicate label color in labels.json: {rgb}")
        color_to_label_id[rgb] = int(label_id_text)

    return color_to_label_id


def _label_name(label_data: object) -> str:
    if isinstance(label_data, dict):
        return str(label_data["name"])
    return str(label_data)


def _validate_contiguous_ids(id2label: dict[int, str]) -> None:
    expected_ids = set(range(len(id2label)))
    actual_ids = set(id2label)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Label ids must be contiguous from 0 to {len(id2label) - 1}. "
            f"Found: {sorted(actual_ids)}"
        )
