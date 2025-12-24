# How to Check Results and Visualizations

After running the quick test or full training, here's how to check your results.

## Quick Check Script

The easiest way is to use the check script:

```bash
# Check latest checkpoint and generate visualizations
python scripts/brats19/check_results.py

# Check specific checkpoint
python scripts/brats19/check_results.py --checkpoint-dir /path/to/checkpoint

# Just list what's available (don't generate visualizations)
python scripts/brats19/check_results.py --list-only

# Customize number of samples
python scripts/brats19/check_results.py --n-samples 20
```

## Manual Check

### 1. Find Your Checkpoint

Checkpoints are saved in:
```
${UVCGAN_OUTDIR}/brats19/model_.../
```

For quick test, look for:
- `model_m(autoencoder)_d(None)_g(vit-unet)_test-pretrain-brats19/` (pretrain)
- `model_m(cyclegan)_d(basic)_g(vit-unet)_test-train-brats19/` (train)

### 2. Check Training History

```bash
# View training history
cat outputs/brats19/model_.../history.csv

# Or in Python
import pandas as pd
df = pd.read_csv('outputs/brats19/model_.../history.csv')
print(df.tail())  # Last few epochs
```

### 3. Generate Visualizations

```bash
# Generate visualizations from checkpoint
python scripts/brats19/eval_and_visualize.py \
    outputs/brats19/model_..._test-train-brats19 \
    --n-samples 10 \
    --split test
```

### 4. View Visualizations

After running eval_and_visualize, check:
```
checkpoint_dir/visualizations/
├── fake_vs_real/        # Translation quality (fake_b vs real_b)
├── fake_a_vs_real_a/    # Reverse translation
└── cycle_consistency_a/ # Cycle consistency
```

## In Colab

### Option 1: Use the Check Script

```python
# Check and visualize results
!python scripts/brats19/check_results.py --n-samples 10
```

### Option 2: Manual Visualization

```python
import os
import glob
from IPython.display import Image, display

# Find latest checkpoint
checkpoint_base = '/content/outputs/brats19'
checkpoints = [d for d in os.listdir(checkpoint_base) if d.startswith('model_m')]
latest = sorted(checkpoints)[-1]
checkpoint_dir = os.path.join(checkpoint_base, latest)

print(f"Using checkpoint: {checkpoint_dir}")

# Generate visualizations
!python scripts/brats19/eval_and_visualize.py \
    {checkpoint_dir} \
    --n-samples 10 \
    --split test

# Display images
vis_dir = os.path.join(checkpoint_dir, 'visualizations', 'fake_vs_real')
if os.path.exists(vis_dir):
    images = sorted(glob.glob(os.path.join(vis_dir, '*.png')))
    print(f"\nFound {len(images)} visualization images:")
    
    for img_path in images[:5]:  # Show first 5
        print(f"\n{os.path.basename(img_path)}:")
        display(Image(img_path))
```

### Option 3: Quick View

```python
# Quick view of what's in the checkpoint
import os
import glob

checkpoint_dir = '/content/outputs/brats19/model_..._test-train-brats19'

# List model files
model_files = glob.glob(os.path.join(checkpoint_dir, 'net_*.pth'))
print(f"Model checkpoints: {len(model_files)}")
for mf in sorted(model_files):
    print(f"  - {os.path.basename(mf)}")

# Check history
history_file = os.path.join(checkpoint_dir, 'history.csv')
if os.path.exists(history_file):
    import pandas as pd
    df = pd.read_csv(history_file)
    print(f"\nTraining history ({len(df)} epochs):")
    print(df.tail())
```

## Understanding the Results

### Directory Structure

```
checkpoint_dir/
├── net_gen_ab_epoch_1.pth    # Generator checkpoints
├── net_gen_ba_epoch_1.pth
├── net_disc_a_epoch_1.pth    # Discriminator checkpoints
├── net_disc_b_epoch_1.pth
├── history.csv                # Training metrics
├── evals/
│   └── test/
│       └── ndarrays_eval-test/
│           ├── fake_a/        # Domain B → A translations
│           ├── fake_b/        # Domain A → B translations
│           ├── real_a/        # Original domain A
│           ├── real_b/        # Original domain B
│           ├── reco_a/        # Cycle reconstruction A
│           └── reco_b/        # Cycle reconstruction B
└── visualizations/
    ├── fake_vs_real/          # Side-by-side comparisons
    ├── fake_a_vs_real_a/
    └── cycle_consistency_a/
```

### What Each Visualization Shows

1. **fake_vs_real** (fake_b vs real_b):
   - Left: Model's translation from domain A to B
   - Right: Ground truth domain B image
   - Shows translation quality

2. **fake_a_vs_real_a** (fake_a vs real_a):
   - Left: Model's translation from domain B to A
   - Right: Ground truth domain A image
   - Shows reverse translation quality

3. **cycle_consistency_a** (reco_a vs real_a):
   - Left: Reconstructed image after A→B→A cycle
   - Right: Original domain A image
   - Shows cycle consistency (how well the model preserves images)

## Quick Commands Summary

```bash
# 1. Check what's available
python scripts/brats19/check_results.py --list-only

# 2. Generate and view visualizations
python scripts/brats19/check_results.py --n-samples 10

# 3. View training history
cat outputs/brats19/model_*/history.csv | tail

# 4. List all checkpoints
ls -la outputs/brats19/model_*/

# 5. Check specific epoch
python scripts/brats19/eval_and_visualize.py \
    outputs/brats19/model_... \
    --epoch 1 \
    --n-samples 5
```

## Troubleshooting

### No visualizations found
- Run `eval_and_visualize.py` first to generate them
- Check that checkpoint directory path is correct

### Images are black/empty
- Check that dataset was loaded correctly
- Verify image transforms are working
- Check image file formats

### Can't find checkpoint
- Check `UVCGAN_OUTDIR` environment variable
- Look in `./outdir/brats19/` if env var not set
- List all: `find . -name "model_*" -type d`

