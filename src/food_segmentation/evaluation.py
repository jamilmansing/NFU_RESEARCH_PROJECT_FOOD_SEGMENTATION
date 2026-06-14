from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from food_segmentation.config import load_label_colors, load_label_maps
from food_segmentation.dataset import SegmentationCsvDataset, segmentation_collate_fn, validate_manifest


def run_test_evaluation(
    model_dir: str | Path,
    manifest_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    batch_size: int = 1,
    image_size: int | None = None,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    validate_data: bool = True,
) -> dict[str, float]:
    """Evaluate a saved SegFormer model on a held-out split and write analysis files."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    id2label, _ = load_label_maps(labels_path)
    color_to_label_id = load_label_colors(labels_path)
    valid_label_ids = set(id2label)

    if validate_data:
        validate_manifest(
            manifest_path=manifest_path,
            splits=(split,),
            valid_label_ids=valid_label_ids,
            color_to_label_id=color_to_label_id,
        )

    processor = AutoImageProcessor.from_pretrained(model_dir)
    if image_size is not None:
        processor.size = {"height": image_size, "width": image_size}
    else:
        image_size = int(processor.size.get("height") or processor.size.get("shortest_edge") or 512)

    model = AutoModelForSemanticSegmentation.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_dataset = SegmentationCsvDataset(
        manifest_path=manifest_path,
        split=split,
        processor=processor,
        valid_label_ids=valid_label_ids,
        color_to_label_id=color_to_label_id,
        image_size=image_size,
        augment=False,
    )

    dataloader_kwargs = {}
    if num_workers > 0:
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=segmentation_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **dataloader_kwargs,
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        num_labels=len(id2label),
        id2label=id2label,
        device=device,
        prefix=split,
    )
    save_json(metrics, output_dir / f"{split}_metrics.json")
    write_per_class_iou(metrics, id2label, output_dir / f"{split}_per_class_iou.csv", prefix=split)
    plot_per_class_iou(metrics, id2label, plots_dir / f"{split}_per_class_iou.png", prefix=split)
    plot_metric_summary(metrics, plots_dir / f"{split}_summary.png", prefix=split)
    return metrics


def evaluate_model(
    model,
    dataloader: DataLoader,
    num_labels: int,
    id2label: dict[int, str],
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            predictions = logits.argmax(dim=1).cpu().numpy()
            label_array = labels.cpu().numpy()
            total_loss += float(outputs.loss.detach().cpu()) * labels.shape[0]
            total_samples += labels.shape[0]
            confusion += batch_confusion_matrix(predictions, label_array, num_labels)

    raw_metrics = metrics_from_confusion(confusion)
    metrics = {
        f"{prefix}_loss": total_loss / max(total_samples, 1),
        f"{prefix}_mean_iou": raw_metrics["mean_iou"],
        f"{prefix}_mean_accuracy": raw_metrics["mean_accuracy"],
        f"{prefix}_overall_accuracy": raw_metrics["overall_accuracy"],
    }
    for label_id, class_iou in enumerate(raw_metrics["per_class_iou"]):
        metrics[f"{prefix}_iou_{metric_name(id2label[label_id])}"] = class_iou
    return metrics


def batch_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_labels: int) -> np.ndarray:
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    valid_mask = (labels >= 0) & (labels < num_labels)
    encoded = num_labels * labels[valid_mask].astype(np.int64) + predictions[valid_mask].astype(np.int64)
    confusion += np.bincount(encoded, minlength=num_labels**2).reshape(num_labels, num_labels)
    return confusion


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | list[float]]:
    intersection = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - intersection
    class_iou = safe_divide(intersection, union)
    class_accuracy = safe_divide(intersection, ground_truth)
    total = confusion.sum()
    return {
        "mean_iou": float(np.nanmean(class_iou)),
        "mean_accuracy": float(np.nanmean(class_accuracy)),
        "overall_accuracy": float(intersection.sum() / total) if total else 0.0,
        "per_class_iou": [float(value) if not np.isnan(value) else 0.0 for value in class_iou],
    }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, fill_value=np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def plot_training_history(run_dir: str | Path, output_dir: str | Path) -> None:
    """Create loss and validation metric plots from Trainer's trainer_state.json."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = find_trainer_state(run_dir)
    if not state_path.exists():
        return

    with state_path.open("r", encoding="utf-8") as file:
        state = json.load(file)
    history = state.get("log_history", [])
    if not history:
        return

    plot_history_series(history, "loss", output_dir / "train_loss.png", "Training Loss")
    plot_history_series(history, "eval_loss", output_dir / "eval_loss.png", "Validation Loss")
    plot_history_series(history, "eval_mean_iou", output_dir / "eval_mean_iou.png", "Validation Mean IoU")
    plot_multi_history_series(
        history,
        ["eval_mean_accuracy", "eval_overall_accuracy"],
        output_dir / "eval_accuracy.png",
        "Validation Accuracy",
    )


def find_trainer_state(run_dir: Path) -> Path:
    root_state = run_dir / "trainer_state.json"
    if root_state.exists():
        return root_state

    checkpoint_states = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: checkpoint_step(path.parent),
    )
    if checkpoint_states:
        return checkpoint_states[-1]
    return root_state


def checkpoint_step(checkpoint_dir: Path) -> int:
    try:
        return int(checkpoint_dir.name.split("-")[-1])
    except ValueError:
        return -1


def plot_history_series(history: list[dict], key: str, output_path: Path, title: str) -> None:
    points = [(entry.get("step"), entry.get(key)) for entry in history if key in entry]
    points = [(step, value) for step, value in points if step is not None and value is not None]
    if not points:
        return
    steps, values = zip(*points)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker="o")
    plt.xlabel("Step")
    plt.ylabel(key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_multi_history_series(
    history: list[dict],
    keys: list[str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    has_points = False
    for key in keys:
        points = [(entry.get("step"), entry.get(key)) for entry in history if key in entry]
        points = [(step, value) for step, value in points if step is not None and value is not None]
        if not points:
            continue
        steps, values = zip(*points)
        plt.plot(steps, values, marker="o", label=key)
        has_points = True
    if not has_points:
        plt.close()
        return
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_metric_summary(metrics: dict[str, float], output_path: Path, prefix: str) -> None:
    keys = [f"{prefix}_mean_iou", f"{prefix}_mean_accuracy", f"{prefix}_overall_accuracy"]
    labels = ["Mean IoU", "Mean Accuracy", "Overall Accuracy"]
    values = [float(metrics.get(key, 0.0)) for key in keys]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=["#2f6f73", "#9b5f2e", "#4c6f9f"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{prefix.title()} Metric Summary")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_per_class_iou(
    metrics: dict[str, float],
    id2label: dict[int, str],
    output_path: Path,
    prefix: str,
) -> None:
    rows = per_class_iou_rows(metrics, id2label, prefix)
    if not rows:
        return
    labels = [row["label"] for row in rows]
    values = [float(row["iou"]) for row in rows]
    height = max(6, min(28, 0.22 * len(rows)))
    plt.figure(figsize=(10, height))
    plt.barh(labels, values, color="#2f6f73")
    plt.xlim(0, 1)
    plt.xlabel("IoU")
    plt.title(f"{prefix.title()} Per-Class IoU")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_per_class_iou(
    metrics: dict[str, float],
    id2label: dict[int, str],
    output_path: Path,
    prefix: str,
) -> None:
    rows = per_class_iou_rows(metrics, id2label, prefix)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label_id", "label", "iou"])
        writer.writeheader()
        writer.writerows(rows)


def per_class_iou_rows(metrics: dict[str, float], id2label: dict[int, str], prefix: str) -> list[dict]:
    rows = []
    for label_id, label in id2label.items():
        metric_key = f"{prefix}_iou_{metric_name(label)}"
        if metric_key not in metrics:
            continue
        rows.append(
            {
                "label_id": label_id,
                "label": label,
                "iou": float(metrics[metric_key]),
            }
        )
    return rows


def metric_name(label: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in label).strip("_")


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
