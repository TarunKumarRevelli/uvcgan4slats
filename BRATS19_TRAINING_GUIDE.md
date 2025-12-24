# Guide: Training UVCGAN on BRATS19 Dataset

This guide outlines the specific changes you need to make to train UVCGAN on your BRATS19 dataset.

## Prerequisites

1. **Dataset Structure**: Organize your BRATS19 dataset as follows:
   ```
   ${UVCGAN_DATA}/brats19/
   ├── train/
   │   ├── domain_a/    (or your domain names, e.g., 't1', 't2', 'flair', 't1ce')
   │   │   └── *.npz files
   │   └── domain_b/
   │       └── *.npz files
   └── test/
       ├── domain_a/
       │   └── *.npz files
       └── domain_b/
           └── *.npz files
   ```

2. **Data Format**: Each `.npz` file should contain a single numpy array. The loader will automatically extract the first array from each `.npz` file.

3. **Image Shape**: Determine your image dimensions (e.g., 256x256, 368x368, etc.)

## Step 1: Set Environment Variables

```bash
export UVCGAN_DATA=/path/to/your/data/directory
export UVCGAN_OUTDIR=/path/to/output/directory
```

## Step 2: Modify Pretraining Script

Edit `scripts/slats/pretrain_slats-256.py` (or create a new script `pretrain_brats19.py`):

### Changes Required:

1. **Update dataset path** (line ~64):
   ```python
   'path'   : 'brats19/',  # Change from 'slats/slats_tiles/'
   ```

2. **Update domain names** (line ~69):
   ```python
   } for domain in [ 'domain_a', 'domain_b' ]  # Change from ['fake', 'real']
   ```
   Replace `domain_a` and `domain_b` with your actual domain names (e.g., `'t1'`, `'t2'` or `'flair'`, `'t1ce'`).

3. **Update image shape** (line ~66):
   ```python
   'shape' : (1, 256, 256),  # Change based on your image size: (channels, height, width)
   ```
   For grayscale MRI images, channels=1. Adjust height and width to match your data.

4. **Add transform if needed** (lines ~67-68):
   ```python
   'transform_train' : custom_transform,  # Add if you need normalization
   'transform_test'  : custom_transform,
   ```
   You can define a transform function similar to the BRaTS example:
   ```python
   def custom_transform(array):
       # Normalize 0-255 data to (-0.5, 0.5)
       return array / 255. - 0.5
   ```

5. **Update label** (line ~111):
   ```python
   'label' : 'pretrain-brats19-256',  # Change from 'pretrain-slats-256'
   ```

6. **Update outdir** (line ~112):
   ```python
   'outdir' : os.path.join(ROOT_OUTDIR, 'brats19'),  # Change from 'slats'
   ```

## Step 3: Modify Training Script

Edit `scripts/slats/train_slats-256.py` (or create a new script `train_brats19.py`):

### Changes Required:

1. **Update dataset path** (line ~101):
   ```python
   'path'   : 'brats19/',  # Change from 'slats/slats_tiles/'
   ```

2. **Update domain names** (line ~106):
   ```python
   } for domain in [ 'domain_a', 'domain_b' ]  # Change from ['fake', 'real']
   ```

3. **Update image shape** (line ~103):
   ```python
   'shape' : (1, 256, 256),  # Match your image dimensions
   ```

4. **Add transform if needed** (lines ~104-105):
   ```python
   'transform_train' : custom_transform,
   'transform_test'  : custom_transform,
   ```

5. **Update transfer configuration** (lines ~154-164):
   If you're using a pretrained model:
   ```python
   'transfer' : {
       'base_model'   : 'brats19/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256',
       'transfer_map' : {
           'gen_ab' : 'encoder',
           'gen_ba' : 'encoder',
       },
       'strict'        : True,
       'allow_partial' : False,
   },
   ```
   If NOT using pretraining, set to `None`:
   ```python
   'transfer' : None,
   ```

6. **Update label** (lines ~166-170):
   ```python
   'label' : (
       f'train-{cmdargs.gen}'
       f'-({cmdargs.lambda_cyc}:{cmdargs.lr_gen}:{cmdargs.lr_disc})'
       '_brats19-256'
   ),
   ```

7. **Update outdir** (line ~171):
   ```python
   'outdir' : os.path.join(ROOT_OUTDIR, 'brats19'),  # Change from 'slats'
   ```

## Step 4: Run Training

### Option A: With Pretraining (Recommended)

1. **Pretrain the generators**:
   ```bash
   python scripts/slats/pretrain_slats-256.py --gen uvcgan --batch_size 64
   ```
   Or if you created a new script:
   ```bash
   python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 64
   ```

2. **Train the translation model**:
   ```bash
   python scripts/slats/train_slats-256.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
   ```

### Option B: Without Pretraining

Skip pretraining and set `'transfer' : None` in the training script, then run:
```bash
python scripts/slats/train_slats-256.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

## Step 5: Hyperparameter Tuning

Consider adjusting these parameters for optimal performance:

1. **Cycle-consistency loss** (`--labmda-cycle`): Default 1.0, try 0.5-2.0
2. **Learning rates** (`--lr-gen`, `--lr-disc`): Default 1e-5 and 5e-5
3. **Gradient penalty** (`--gp-constant`, `--gp-lambda`): Default 10.0 and 1.0
4. **Batch size**: Adjust based on GPU memory
5. **Image shape**: Ensure it matches your actual data dimensions

## Example: Complete Modified Configuration

Here's an example configuration for BRATS19 with 256x256 images:

```python
'data' : {
    'datasets' : [
        {
            'dataset' : {
                'name'   : 'ndarray-domain-hierarchy',
                'domain' : domain,
                'path'   : 'brats19/',
            },
            'shape'           : (1, 256, 256),
            'transform_train' : custom_transform,  # Optional
            'transform_test'  : custom_transform,  # Optional
        } for domain in [ 't1', 't2' ]  # Example domain names
    ],
    'merge_type' : 'unpaired',
},
```

## Notes

- The `ndarray-domain-hierarchy` dataset loader automatically finds all `.npz` files in the specified directories
- Each `.npz` file should contain a single numpy array (the loader uses the first array found)
- Make sure your image dimensions match the `shape` parameter
- If your data needs normalization, use the `transform_train` and `transform_test` parameters
- The pretraining step uses BERT-like masking (40% of 32x32 patches masked) which works well for medical images

## Troubleshooting

1. **File not found errors**: Check that `UVCGAN_DATA` is set correctly and your dataset path matches
2. **Shape mismatches**: Verify your image dimensions match the `shape` parameter
3. **Memory errors**: Reduce batch size or image dimensions
4. **Empty dataset**: Ensure `.npz` files are in the correct directory structure

