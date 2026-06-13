from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from food_segmentation.config import load_label_colors, load_label_maps
from food_segmentation.dataset import (
    SegmentationCsvDataset,
    segmentation_collate_fn,
    validate_manifest,
)


MIT_B0_CHECKPOINT = "nvidia/mit-b0"
DEFAULT_OUTPUT_DIR = "outputs/segformer-mit-b0-food"


def run_training_smoke_test(
    manifest_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    checkpoint: str = MIT_B0_CHECKPOINT,
    max_steps: int = 5,
    batch_size: int = 2,
) -> None:
    """Run a tiny SegFormer training pass to validate the custom-data pipeline."""
    id2label, label2id = load_label_maps(labels_path)
    color_to_label_id = load_label_colors(labels_path)
    valid_label_ids = set(id2label)

    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_reduce_labels=False,
    )
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )

    train_dataset = SegmentationCsvDataset(
        manifest_path=manifest_path,
        split="train",
        processor=processor,
        valid_label_ids=valid_label_ids,
        color_to_label_id=color_to_label_id,
    )
    eval_dataset = SegmentationCsvDataset(
        manifest_path=manifest_path,
        split="val",
        processor=processor,
        valid_label_ids=valid_label_ids,
        color_to_label_id=color_to_label_id,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=max_steps,
        save_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=segmentation_collate_fn,
    )

    trainer.train()
    trainer.evaluate()


def run_full_training(
    manifest_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint: str = MIT_B0_CHECKPOINT,
    epochs: int = 50,
    batch_size: int = 2,
    eval_batch_size: int = 1,
    learning_rate: float = 6e-5,
    image_size: int = 512,
    weight_decay: float = 0.01,
    resume_from_checkpoint: str | Path | None = None,
    logging_steps: int = 10,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> Path:
    """Train SegFormer with a MiT-B0 backbone and save a deployable HF model."""
    output_dir = Path(output_dir)
    final_dir = output_dir / "final"

    print("Loading labels...", flush=True)
    id2label, label2id = load_label_maps(labels_path)
    color_to_label_id = load_label_colors(labels_path)
    valid_label_ids = set(id2label)
    validate_manifest(
        manifest_path=manifest_path,
        splits=("train", "val"),
        valid_label_ids=valid_label_ids,
        color_to_label_id=color_to_label_id,
    )

    print(f"Loading processor and model from {checkpoint}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_reduce_labels=False,
    )
    processor.size = {"height": image_size, "width": image_size}
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )

    print("Building train and validation datasets...", flush=True)
    train_dataset = SegmentationCsvDataset(
        manifest_path=manifest_path,
        split="train",
        processor=processor,
        valid_label_ids=valid_label_ids,
        color_to_label_id=color_to_label_id,
        image_size=image_size,
        augment=True,
    )
    eval_dataset = SegmentationCsvDataset(
        manifest_path=manifest_path,
        split="val",
        processor=processor,
        valid_label_ids=valid_label_ids,
        color_to_label_id=color_to_label_id,
        image_size=image_size,
        augment=False,
    )
    print(
        "Starting training: "
        f"{len(train_dataset)} train samples, {len(eval_dataset)} val samples, "
        f"{epochs} epochs, train batch size {batch_size}, "
        f"eval batch size {eval_batch_size}, image size {image_size}, "
        f"num workers {num_workers}."
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_accumulation_steps=1,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        greater_is_better=True,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=logging_steps,
        disable_tqdm=False,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=persistent_workers if num_workers > 0 else False,
        dataloader_prefetch_factor=prefetch_factor if num_workers > 0 else None,
        remove_unused_columns=False,
        report_to=[],
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=segmentation_collate_fn,
        compute_metrics=build_compute_metrics(num_labels=len(id2label), id2label=id2label),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[ConsoleProgressCallback(logging_steps=logging_steps)],
    )

    print("Trainer is starting. If tqdm does not render, step logs will still print.", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training complete. Running final evaluation...", flush=True)
    eval_metrics = trainer.evaluate()

    print(f"Saving final model to {final_dir}...", flush=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    _copy_labels(labels_path, final_dir)
    _save_json(train_result.metrics, final_dir / "train_metrics.json")
    _save_json(eval_metrics, final_dir / "eval_metrics.json")

    return final_dir


class ConsoleProgressCallback(TrainerCallback):
    def __init__(self, logging_steps: int) -> None:
        self.logging_steps = max(logging_steps, 1)

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Train begin: max_steps={state.max_steps}, epochs={args.num_train_epochs}.", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1 or state.global_step % self.logging_steps == 0:
            print(f"Training step {state.global_step}/{state.max_steps} complete.", flush=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch or 0:.2f} started.", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch or 0:.2f} ended.", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            mean_iou = metrics.get("eval_mean_iou")
            eval_loss = metrics.get("eval_loss")
            print(f"Evaluation complete: eval_loss={eval_loss}, eval_mean_iou={mean_iou}.", flush=True)

    def on_save(self, args, state, control, **kwargs):
        print(f"Checkpoint saved at step {state.global_step}.", flush=True)


def build_compute_metrics(num_labels: int, id2label: dict[int, str]):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        predictions = np.asarray(predictions)
        labels_array = np.asarray(labels)

        metrics = segmentation_metrics(
            predictions=predictions,
            labels=labels_array,
            num_labels=num_labels,
        )
        for label_id, class_iou in enumerate(metrics.pop("per_class_iou")):
            metrics[f"iou_{_metric_name(id2label[label_id])}"] = class_iou
        return metrics

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """Store class predictions instead of full float logits during evaluation."""
    if isinstance(logits, tuple):
        logits = logits[0]

    resized_logits = F.interpolate(
        logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return resized_logits.argmax(dim=1)


def segmentation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
) -> dict[str, float | list[float]]:
    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)
    valid_mask = (labels >= 0) & (labels < num_labels)
    encoded = num_labels * labels[valid_mask].astype(np.int64) + predictions[valid_mask].astype(np.int64)
    confusion += np.bincount(encoded, minlength=num_labels**2).reshape(num_labels, num_labels)

    intersection = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - intersection

    class_iou = _safe_divide(intersection, union)
    class_accuracy = _safe_divide(intersection, ground_truth)
    total = confusion.sum()

    return {
        "mean_iou": float(np.nanmean(class_iou)),
        "mean_accuracy": float(np.nanmean(class_accuracy)),
        "overall_accuracy": float(intersection.sum() / total) if total else 0.0,
        "per_class_iou": [float(value) if not np.isnan(value) else 0.0 for value in class_iou],
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, fill_value=np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _copy_labels(labels_path: str | Path, final_dir: Path) -> None:
    shutil.copy2(labels_path, final_dir / "labels.json")


def _save_json(data: dict[str, float], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _metric_name(label: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in label).strip("_")
