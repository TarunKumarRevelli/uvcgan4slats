# Quick Fix: translate_data.py Argument Error

## The Issue

The error `unrecognized arguments: --n-eval 10` occurs because `translate_data.py` uses `-n` (short form), not `--n-eval`.

## Correct Usage

### translate_data.py

```bash
# Correct - use -n (short form)
python scripts/translate_data.py \
    /path/to/checkpoint \
    -n 10 \
    --split test

# Wrong - don't use --n-eval
python scripts/translate_data.py \
    /path/to/checkpoint \
    --n-eval 10  # ❌ This will fail
```

### eval_and_visualize.py

This script already uses the correct `-n` argument internally, so it should work:

```bash
python scripts/brats19/eval_and_visualize.py \
    /path/to/checkpoint \
    --n-samples 10 \
    --split test
```

## All Available Arguments for translate_data.py

```bash
python scripts/translate_data.py --help
```

Common arguments:
- `MODEL` (required): Path to checkpoint directory
- `-n N`: Number of samples to evaluate
- `--split {train,test,val}`: Dataset split (default: test)
- `--epoch EPOCH`: Specific epoch to evaluate (default: latest)
- `--model-state {train,eval}`: Model state (default: eval)
- `--batch-size BATCH_SIZE`: Batch size (default: 1)

## Example Commands

```bash
# Evaluate latest checkpoint, 10 samples
python scripts/translate_data.py outputs/brats19/model_... -n 10

# Evaluate specific epoch
python scripts/translate_data.py outputs/brats19/model_... -n 10 --epoch 1

# Evaluate training set
python scripts/translate_data.py outputs/brats19/model_... -n 20 --split train
```

## If You're Still Getting Errors

1. Make sure you're using `-n` not `--n-eval`
2. Check that the checkpoint path is correct
3. Verify the checkpoint directory exists and contains model files

