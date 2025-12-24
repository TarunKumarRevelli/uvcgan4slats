# BRATS19 Visualization Guide

## Overview

This guide explains how to visualize UVCGAN training results and monitor model performance during training.

## Changes Made

### 1. Checkpoint Frequency
- **Pretraining**: Changed from every 100 epochs to **every epoch** (`checkpoint: 1`)
- **Training**: Changed from every 50 epochs to **every epoch** (`checkpoint: 1`)

This means you'll get a checkpoint after each epoch, allowing you to monitor progress more closely.

### 2. Visualization Scripts

Two new scripts have been created:

#### `visualize_results.py`
Creates side-by-side comparison images from translation results.

#### `eval_and_visualize.py`
Complete workflow: evaluates a checkpoint, generates translations, and creates visualizations.

## Usage

### Option 1: Quick Visualization (After Training)

After training completes, visualize the final results:

```bash
# Evaluate and visualize the latest checkpoint
python scripts/brats19/eval_and_visualize.py \
    /path/to/checkpoint/directory \
    --n-samples 20 \
    --split test
```

This will:
1. Generate translations for test set
2. Create side-by-side comparisons (fake vs real)
3. Create cycle consistency visualizations
4. Save all plots to `checkpoint_dir/visualizations/`

### Option 2: Visualize Specific Epoch

To visualize a specific epoch during training:

```bash
python scripts/brats19/eval_and_visualize.py \
    /path/to/checkpoint/directory \
    --epoch 10 \
    --n-samples 10 \
    --split test
```

### Option 3: Manual Visualization

If you already have translation results:

```bash
# Compare fake_b (translations) with real_b (ground truth)
python scripts/brats19/visualize_results.py \
    /path/to/results/fake_b \
    /path/to/output/visualizations \
    --n-samples 20 \
    --compare-with /path/to/results/real_b \
    --layout horizontal
```

## Directory Structure

After running `eval_and_visualize.py`, you'll have:

```
checkpoint_dir/
├── evals/
│   └── test/
│       └── ndarrays_eval-test/
│           ├── fake_a/      # Domain A → Domain B translations
│           ├── fake_b/      # Domain B → Domain A translations
│           ├── real_a/      # Original Domain A images
│           ├── real_b/      # Original Domain B images
│           ├── reco_a/      # Cycle reconstruction A
│           └── reco_b/      # Cycle reconstruction B
└── visualizations/
    ├── fake_vs_real/       # fake_b vs real_b comparisons
    ├── fake_a_vs_real_a/   # fake_a vs real_a comparisons
    └── cycle_consistency_a/ # reco_a vs real_a (cycle consistency)
```

## Visualization Types

### 1. Translation Quality (fake_b vs real_b)
Shows how well the model translates from domain A to domain B.

### 2. Reverse Translation (fake_a vs real_a)
Shows how well the model translates from domain B to domain A.

### 3. Cycle Consistency (reco_a vs real_a)
Shows how well the model reconstructs images after a round trip (A→B→A).

## Colab Usage

Add this cell to your Colab notebook after training:

```python
# Visualize results from latest checkpoint
import os

checkpoint_dir = '/content/outputs/brats19/model_m(cyclegan)_d(basic)_g(vit-unet)_train-uvcgan-(1.0:1e-05:5e-05)_brats19-256'

!python scripts/brats19/eval_and_visualize.py \
    {checkpoint_dir} \
    --n-samples 10 \
    --split test

# Display images
from IPython.display import Image, display
import glob

vis_dir = os.path.join(checkpoint_dir, 'visualizations', 'fake_vs_real')
images = sorted(glob.glob(os.path.join(vis_dir, '*.png')))

for img_path in images[:5]:  # Show first 5
    display(Image(img_path))
```

## Monitoring Training Progress

### During Training

You can monitor training by checking the checkpoint directory:

```bash
# List all checkpoints
ls /path/to/checkpoint/directory/model_epoch_*.pth

# Check training history
cat /path/to/checkpoint/directory/history.csv
```

### After Each Epoch

To visualize results after each epoch (e.g., epoch 5):

```bash
python scripts/brats19/eval_and_visualize.py \
    /path/to/checkpoint/directory \
    --epoch 5 \
    --n-samples 5
```

## Tips

1. **Start Small**: Use `--n-samples 5` for quick checks during training
2. **Full Evaluation**: Use `--n-samples 50` or more for final evaluation
3. **Save to Drive**: In Colab, copy visualizations to Drive:
   ```python
   !cp -r /content/outputs/.../visualizations /content/drive/MyDrive/
   ```
4. **Compare Epochs**: Visualize multiple epochs to see improvement:
   ```bash
   for epoch in 1 10 20 50; do
       python scripts/brats19/eval_and_visualize.py \
           checkpoint_dir --epoch $epoch --n-samples 5
   done
   ```

## Troubleshooting

### No images found
- Make sure you've run `translate_data.py` first, or use `eval_and_visualize.py` which does it automatically

### Checkpoint not found
- Check the checkpoint directory path
- Verify the epoch number exists: `ls checkpoint_dir/model_epoch_*.pth`

### Visualization errors
- Ensure PIL/Pillow is installed: `pip install pillow`
- Check that image files are readable

## Example Workflow

```bash
# 1. Train model (checkpoints saved every epoch)
python scripts/brats19/train_brats19.py --gen uvcgan

# 2. After training, visualize epoch 10
python scripts/brats19/eval_and_visualize.py \
    outputs/brats19/model_..._train-... \
    --epoch 10 \
    --n-samples 20

# 3. Visualize final model (latest epoch)
python scripts/brats19/eval_and_visualize.py \
    outputs/brats19/model_..._train-... \
    --n-samples 50

# 4. View results
ls outputs/brats19/model_.../visualizations/
```

