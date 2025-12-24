# Quick Start: Training UVCGAN on BRATS19

## Summary of Changes Required

Based on the README and codebase analysis, here are the **essential changes** you need to make:

## 1. Dataset Organization

Organize your BRATS19 dataset in this structure:
```
${UVCGAN_DATA}/brats19/
├── train/
│   ├── domain_a/    # Replace with your domain names (e.g., 't1', 't2', 'flair', 't1ce')
│   │   └── *.npz files
│   └── domain_b/
│       └── *.npz files
└── test/
    ├── domain_a/
    │   └── *.npz files
    └── domain_b/
        └── *.npz files
```

## 2. Set Environment Variables

```bash
export UVCGAN_DATA=/path/to/your/data/directory
export UVCGAN_OUTDIR=/path/to/output/directory
```

## 3. Modify Training Scripts

I've created example scripts for you in `scripts/brats19/`. You need to edit these files and update the configuration section:

### In `scripts/brats19/pretrain_brats19.py`:
- **Line 50**: `DATASET_PATH = 'brats19/'` - Keep as is if your data is in `${UVCGAN_DATA}/brats19/`
- **Line 51**: `DOMAIN_NAMES = ['domain_a', 'domain_b']` - **CHANGE THIS** to your actual domain names
- **Line 52**: `IMAGE_SHAPE = (1, 256, 256)` - **CHANGE THIS** to match your image dimensions
- **Line 60**: `USE_TRANSFORM = False` - Set to `True` if you need normalization

### In `scripts/brats19/train_brats19.py`:
- **Line 78**: `DATASET_PATH = 'brats19/'` - Keep as is
- **Line 79**: `DOMAIN_NAMES = ['domain_a', 'domain_b']` - **CHANGE THIS** to match pretraining script
- **Line 80**: `IMAGE_SHAPE = (1, 256, 256)` - **CHANGE THIS** to match pretraining script
- **Line 88**: `USE_TRANSFORM = False` - Set to `True` if you need normalization

## 4. Run Training

### Step 1: Pretrain (Recommended)
```bash
python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 64
```

### Step 2: Train Translation Model
```bash
python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

If you want to train without pretraining:
```bash
python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5 --no-pretrain
```

## Key Configuration Parameters

| Parameter | Location | What to Change |
|-----------|----------|----------------|
| Dataset path | Both scripts | `DATASET_PATH` - relative to `${UVCGAN_DATA}` |
| Domain names | Both scripts | `DOMAIN_NAMES` - your two domains (e.g., `['t1', 't2']`) |
| Image shape | Both scripts | `IMAGE_SHAPE` - `(channels, height, width)` |
| Transform | Both scripts | `USE_TRANSFORM` - set to `True` if normalization needed |

## Example: T1 to T2 Translation

If you want to translate from T1 to T2 images:

1. **Dataset structure**:
   ```
   ${UVCGAN_DATA}/brats19/
   ├── train/
   │   ├── t1/
   │   └── t2/
   └── test/
       ├── t1/
       └── t2/
   ```

2. **Configuration**:
   ```python
   DOMAIN_NAMES = ['t1', 't2']
   IMAGE_SHAPE = (1, 256, 256)  # Adjust to your image size
   ```

## Troubleshooting

- **File not found**: Check `UVCGAN_DATA` environment variable
- **Shape mismatch**: Verify `IMAGE_SHAPE` matches your actual data
- **Empty dataset**: Ensure `.npz` files are in correct folders
- **Memory errors**: Reduce `--batch_size` in pretraining script

For more detailed information, see `BRATS19_TRAINING_GUIDE.md`.

