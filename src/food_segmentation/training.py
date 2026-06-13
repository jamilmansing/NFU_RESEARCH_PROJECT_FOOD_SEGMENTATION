from __future__ import annotations

from pathlib import Path

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)

from food_segmentation.config import load_label_colors, load_label_maps
from food_segmentation.dataset import SegmentationCsvDataset, segmentation_collate_fn


MIT_B0_CHECKPOINT = "nvidia/mit-b0"


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
