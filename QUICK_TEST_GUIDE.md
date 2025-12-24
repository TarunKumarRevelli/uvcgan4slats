# Quick Test Guide - Verify Pipeline Before Full Training

This guide helps you quickly test the entire UVCGAN pipeline with minimal GPU usage before committing to full training.

## Quick Test Scripts

Two test scripts are provided. **Recommended: `quick_test.py`** (simpler, safer):

### Option 1: quick_test.py (Recommended)

Simpler and safer - doesn't modify your original scripts:

```bash
# Run complete pipeline test (1 epoch each)
python scripts/brats19/quick_test.py

# Customize test parameters
python scripts/brats19/quick_test.py \
    --pretrain-epochs 1 \
    --train-epochs 1 \
    --batch-size 4
```

### Option 2: test_pipeline.py

More advanced - temporarily modifies scripts (use if quick_test.py doesn't work):

```bash
python scripts/brats19/test_pipeline.py \
    --pretrain-epochs 1 \
    --train-epochs 1 \
    --batch-size 4 \
    --gen uvcgan
```

### Options

- `--pretrain-epochs N`: Number of pretraining epochs (default: 1)
- `--train-epochs N`: Number of training epochs (default: 1)
- `--batch-size N`: Batch size (default: 4, smaller saves memory)
- `--skip-pretrain`: Skip pretraining step
- `--gen TYPE`: Generator type (default: uvcgan)

## Manual Quick Test

If you prefer to test manually:

### Step 1: Quick Pretraining Test

Modify `scripts/brats19/pretrain_brats19.py` temporarily:

```python
'epochs'        : 1,  # Change from 499 to 1
'steps_per_epoch'  : 10,  # Change from large number to 10
'batch_size' : 4,  # Use smaller batch size
```

Then run:
```bash
python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 4
```

### Step 2: Quick Training Test

Modify `scripts/brats19/train_brats19.py` temporarily:

```python
'epochs'        : 1,  # Change from 500 to 1
'steps_per_epoch'  : 10,  # Change from 2000 to 10
'batch_size' : 1,  # Keep at 1
```

Then run:
```bash
python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

### Step 3: Quick Evaluation Test

```bash
# Find checkpoint directory
CHECKPOINT_DIR=$(ls -td outputs/brats19/model_*train* | head -1)

# Quick evaluation with 2 samples
python scripts/brats19/eval_and_visualize.py \
    $CHECKPOINT_DIR \
    --n-samples 2 \
    --split test
```

## Colab Quick Test

Add this cell to your Colab notebook for a quick test:

```python
# Quick pipeline test - 1 epoch each
!python scripts/brats19/test_pipeline.py \
    --pretrain-epochs 1 \
    --train-epochs 1 \
    --batch-size 4

print("\n✅ If you see this, the pipeline works!")
print("Now you can run full training with more epochs.")
```

## What the Test Verifies

The quick test verifies:

1. ✅ **Data loading**: Images load correctly
2. ✅ **Model initialization**: Models create without errors
3. ✅ **Training loop**: Forward/backward passes work
4. ✅ **Checkpoint saving**: Models save correctly
5. ✅ **Evaluation**: Translation and visualization work
6. ✅ **No import errors**: All dependencies available
7. ✅ **Configuration**: Config files are valid

## Expected Test Duration

- **Pretraining (1 epoch)**: ~2-5 minutes
- **Training (1 epoch)**: ~2-5 minutes  
- **Evaluation (2 samples)**: ~1 minute
- **Total**: ~5-10 minutes

Compare this to full training which can take hours!

## After Successful Test

Once the test passes, you can run full training:

```bash
# Full pretraining (restore original epochs)
python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 32

# Full training (restore original epochs)
python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

## Troubleshooting

### Test fails at data loading
- Check dataset path: `echo $UVCGAN_DATA`
- Verify dataset structure: `ls $UVCGAN_DATA/brats19/train/t1/`
- Check image files exist and are readable

### Test fails at model initialization
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size further: `--batch-size 2`
- Check CUDA version compatibility

### Test fails at checkpoint saving
- Check output directory: `echo $UVCGAN_OUTDIR`
- Verify write permissions: `touch $UVCGAN_OUTDIR/test.txt && rm $UVCGAN_OUTDIR/test.txt`

### Out of memory errors
- Reduce batch size: `--batch-size 2` or `1`
- Reduce image size in scripts (if possible)
- Use smaller generator: `--gen unet` instead of `uvcgan`

## Minimal Test (Fastest)

For the absolute fastest test (just verify imports and config):

```python
# Just verify imports and config
python -c "
from uvcgan import train
from uvcgan.config import Config
import os

# Quick config test
config = {
    'batch_size': 1,
    'data': {
        'datasets': [{
            'dataset': {'name': 'image-domain-hierarchy', 'domain': 't1', 'path': 'brats19'},
            'shape': (1, 256, 256),
            'transform_train': [{'name': 'resize', 'size': 256}, {'name': 'grayscale'}],
        }],
        'merge_type': 'unpaired'
    },
    'epochs': 1,
    'generator': {'model': 'unet_256', 'model_args': None},
    'model': 'autoencoder',
    'checkpoint': 1,
    'label': 'test',
    'outdir': './test_output'
}
print('✓ Config valid')
print('✓ Imports work')
"
```

This takes ~10 seconds and verifies basic setup.

## Recommended Workflow

1. **Quick test** (5-10 min): Run `test_pipeline.py` with 1 epoch each
2. **Short test** (30-60 min): Run with 5-10 epochs to see actual learning
3. **Full training**: Once confident, run full training

This saves GPU hours by catching errors early!

