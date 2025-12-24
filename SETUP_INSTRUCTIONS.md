# Setup Instructions for BRATS19 Training

## Step 1: Create Conda Environment

Yes, you should create a conda environment based on `conda_env.yml`. This ensures you have all the required dependencies (PyTorch, CUDA, etc.) properly configured.

```bash
# From the uvcgan4slats source folder
conda env create -f contrib/conda_env.yml
```

This will create a conda environment named `uvcgan4slats` with Python 3.7, PyTorch 1.9.0, CUDA 11.1, and all other dependencies.

## Step 2: Activate the Environment

```bash
conda activate uvcgan4slats
```

## Step 3: Install uvcgan4slats Package

```bash
# Make sure you're in the uvcgan4slats source folder
python setup.py develop --user
```

**Note**: If you run this with `sudo`, remove the `--user` flag.

## Step 4: Install toytools Dependency

The `uvcgan4slats` package depends on `toytools`. Install it:

```bash
git clone https://github.com/LS4GAN/toytools
cd toytools
python setup.py develop --user
cd ..
```

**Note**: If you run this with `sudo`, remove the `--user` flag.

## Step 5: Set Environment Variables

Set the data and output directories:

```bash
export UVCGAN_DATA=/path/to/your/data/directory
export UVCGAN_OUTDIR=/path/to/output/directory
```

For example:
```bash
export UVCGAN_DATA=/home/tarun-kumar-revelli/Desktop/FYP/data
export UVCGAN_OUTDIR=/home/tarun-kumar-revelli/Desktop/FYP/outputs
```

## Step 6: Verify Dataset Structure

Make sure your BRATS19 dataset is organized as:
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

## Step 7: Ready to Train!

Now you can proceed with training:

### Pretraining (Recommended):
```bash
python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 64
```

### Main Training:
```bash
python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

## Troubleshooting

### If conda environment creation fails:
- Make sure you have conda/miniconda installed
- Try updating conda: `conda update conda`
- Check if you have enough disk space

### If CUDA issues occur:
- Verify your GPU and CUDA drivers are compatible with CUDA 11.1
- The environment uses PyTorch 1.9.0 with CUDA 11.1
- You can check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### If import errors occur:
- Make sure you activated the conda environment: `conda activate uvcgan4slats`
- Verify installation: `python -c "import uvcgan; print('OK')"`
- Check that toytools is installed: `python -c "import toytools; print('OK')"`

## Quick Checklist

- [ ] Conda environment created (`conda env create -f contrib/conda_env.yml`)
- [ ] Environment activated (`conda activate uvcgan4slats`)
- [ ] uvcgan4slats installed (`python setup.py develop --user`)
- [ ] toytools installed (`python setup.py develop --user` in toytools folder)
- [ ] Environment variables set (`UVCGAN_DATA` and `UVCGAN_OUTDIR`)
- [ ] Dataset organized correctly (brats19/train/t1/, brats19/train/t2/, etc.)
- [ ] Ready to train!

