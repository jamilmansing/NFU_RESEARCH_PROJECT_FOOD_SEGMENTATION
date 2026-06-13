# NFU Food Segmentation

SegFormer scaffold for two workflows:

- demo inference with `nvidia/segformer-b0-finetuned-ade-512-512`
- custom food-segmentation training with the `nvidia/mit-b0` backbone

## Environment Check

```powershell
uv run python main.py
```

## Demo Inference

This uses the ADE20K-finetuned checkpoint only as a quick sanity check. Its labels are ADE20K labels, not your food labels.

```powershell
uv run python scripts/run_inference.py --image path/to/image.jpg
```

The default output is:

```text
outputs/inference/ade20k_overlay.png
```

## Custom Training Smoke Test

Create a CSV manifest with this format:

```csv
image_path,mask_path,split
data/images/sample_001.jpg,data/masks/sample_001.png,train
data/images/sample_002.jpg,data/masks/sample_002.png,val
```

Then run:

```powershell
uv run python scripts/train_smoke_test.py --manifest path/to/manifest.csv
```

Masks can be either single-channel class-id images or RGB color masks.

For class-id masks, each pixel value is the label id from `configs/labels.json`.

For RGB masks, `configs/labels.json` must include a `color` for each label:

```json
{
  "0": {
    "name": "background",
    "color": [0, 0, 0]
  },
  "1": {
    "name": "bawan",
    "color": [119, 185, 241]
  }
}
```

The loader converts each RGB mask pixel to the matching label id before training.

Class `0` is treated as a real background class, so the processor uses `do_reduce_labels=False`.

## Full Training

Run full epoch-based training with MiT-B0 as the SegFormer backbone:

```powershell
uv run python scripts/train.py --manifest data/manifest.csv --labels configs/labels.json
```

Useful options:

```powershell
uv run python scripts/train.py `
  --manifest data/manifest.csv `
  --labels configs/labels.json `
  --output-dir outputs/segformer-mit-b0-food `
  --epochs 50 `
  --batch-size 2 `
  --learning-rate 6e-5 `
  --image-size 512 `
  --weight-decay 0.01
```

The training script saves checkpoints every epoch, keeps the best checkpoint by validation `mean_iou`, and writes the final Hugging Face model here:

```text
outputs/segformer-mit-b0-food/final
```

Resume from a Trainer checkpoint:

```powershell
uv run python scripts/train.py --manifest data/manifest.csv --labels configs/labels.json --resume-from-checkpoint outputs/segformer-mit-b0-food/checkpoint-100
```

## ONNX Runtime

Export the ADE20K demo model:

```powershell
uv run python scripts/export_onnx.py --kind ade20k --output outputs/onnx/segformer_ade20k.onnx
```

Run ONNX Runtime inference:

```powershell
uv run python scripts/run_onnx_inference.py --onnx outputs/onnx/segformer_ade20k.onnx --image path/to/image.jpg
```

Export a MIT-B0 model with your custom food labels:

```powershell
uv run python scripts/export_onnx.py --kind mit-b0 --labels configs/labels.json --output outputs/onnx/segformer_mit_b0_food.onnx
```

That MIT-B0 export is only a pipeline check until you export a trained checkpoint. After training, pass your trained model directory with `--checkpoint`.

Export a trained model:

```powershell
uv run python scripts/export_onnx.py --kind mit-b0 --checkpoint outputs/segformer-mit-b0-food/final --labels configs/labels.json --output outputs/onnx/food_segformer.onnx
```
