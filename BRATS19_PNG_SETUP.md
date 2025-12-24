# BRATS19 PNG Dataset Setup

## Important Changes Made

Since your BRATS19 dataset contains **PNG images** (not `.npz` files), the scripts have been updated to use the `image-domain-hierarchy` dataset loader instead of `ndarray-domain-hierarchy`.

## Dataset Structure

Your dataset should be organized as:
```
${UVCGAN_DATA}/brats19/
├── train/
│   ├── t1/
│   │   └── *.png files
│   └── t2/
│       └── *.png files
└── test/
    ├── t1/
    │   └── *.png files
    └── t2/
        └── *.png files
```

## Configuration in Scripts

Both `pretrain_brats19.py` and `train_brats19.py` are now configured with:

1. **Dataset Type**: `image-domain-hierarchy` (for PNG/image files)
2. **Dataset Path**: `brats19` (relative to `${UVCGAN_DATA}`)
3. **Domain Names**: `['t1', 't2']`
4. **Image Shape**: `(1, 256, 256)` - assumes grayscale images
5. **Transforms**: 
   - `resize` to 256x256 (ensures all images are the same size)
   - Automatic conversion to tensors (values normalized to 0-1)

## Important Notes

### Image Channels

- **If your PNGs are grayscale** (1 channel): Keep `IMAGE_SHAPE = (1, 256, 256)`
- **If your PNGs are RGB** (3 channels): Change to `IMAGE_SHAPE = (3, 256, 256)`

The `image-domain-hierarchy` loader will automatically:
- Load PNG files using PIL/torchvision
- Convert them to tensors (values 0-1)
- Apply the specified transforms (resize, etc.)

### Transform Behavior

- Images are automatically converted to tensors with values in range [0, 1]
- The `resize` transform ensures all images are resized to `TARGET_SIZE` (256x256)
- You can add data augmentation transforms like `random-flip-horizontal` if needed

## Running Training

### Step 1: Pretrain
```bash
python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 64
```

### Step 2: Train Translation Model
```bash
python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

## Troubleshooting

1. **Shape mismatch errors**: 
   - Check if your PNGs are RGB (3 channels) or grayscale (1 channel)
   - Update `IMAGE_SHAPE` accordingly: `(3, 256, 256)` for RGB or `(1, 256, 256)` for grayscale

2. **File not found**:
   - Verify `UVCGAN_DATA` environment variable is set correctly
   - Ensure PNG files are in `brats19/train/t1/`, `brats19/train/t2/`, etc.

3. **Different image sizes**:
   - The `resize` transform will automatically resize all images to `TARGET_SIZE` (256x256)
   - If you need a different size, change `TARGET_SIZE` and update `IMAGE_SHAPE` accordingly

4. **Memory errors**:
   - Reduce `--batch_size` in the pretraining script
   - Reduce image size if possible

## Differences from NPZ Setup

| Aspect | NPZ Files | PNG Files |
|--------|-----------|-----------|
| Dataset loader | `ndarray-domain-hierarchy` | `image-domain-hierarchy` |
| File format | `.npz` (numpy arrays) | `.png` (image files) |
| Data loading | Direct numpy array loading | PIL/torchvision image loading |
| Normalization | Manual (via custom transform) | Automatic (ToTensor: 0-1 range) |
| Shape handling | Must match array shape exactly | Resize transform handles different sizes |

The scripts are now ready to work with your PNG images!

