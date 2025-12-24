# Training UVCGAN on Free Cloud GPUs

This guide covers how to train UVCGAN on free cloud GPU platforms. The two best options are **Google Colab** and **Kaggle Notebooks**.

## Option 1: Google Colab (Recommended)

### Advantages:
- Free GPU (T4, P100, or V100) with ~12-15 hours runtime
- Easy to use, no signup complexity
- Can mount Google Drive for persistent storage
- Good for interactive development

### Setup Steps:

#### 1. Create a Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → Hardware accelerator: GPU

#### 2. Install Dependencies

Run these cells in your Colab notebook:

```python
# Install conda in Colab (if needed)
!pip install condacolab
import condacolab
condacolab.install()

# Restart runtime after this cell
```

After restart, continue:

```python
# Install system dependencies
!apt-get update
!apt-get install -y git

# Clone the repository
!git clone https://github.com/LS4GAN/uvcgan4slats.git
%cd uvcgan4slats

# Create conda environment (simplified - install packages directly)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pillow matplotlib tqdm

# Install toytools
!git clone https://github.com/LS4GAN/toytools
%cd toytools
!pip install -e .
%cd ..

# Install uvcgan4slats
!pip install -e .
```

#### 3. Upload Dataset to Google Drive

**Option A: Upload via Colab UI**
```python
from google.colab import files
uploaded = files.upload()  # Upload your brats19 folder as zip, then extract
```

**Option B: Mount Google Drive (Recommended)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy your dataset to Colab workspace
!cp -r /content/drive/MyDrive/brats19 /content/brats19
```

#### 4. Set Environment Variables

```python
import os
os.environ['UVCGAN_DATA'] = '/content'
os.environ['UVCGAN_OUTDIR'] = '/content/outputs'
```

#### 5. Run Training

```python
# Pretraining
!python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 32

# Main training
!python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

#### 6. Save Results

```python
# Save outputs to Google Drive
!cp -r /content/outputs /content/drive/MyDrive/uvcgan_outputs
```

### Colab Limitations:
- Runtime disconnects after ~12 hours of inactivity
- GPU availability is not guaranteed (may need to wait)
- Limited disk space (~80GB)

---

## Option 2: Kaggle Notebooks

### Advantages:
- Free GPU (P100) with 30 hours/week
- More stable than Colab
- Built-in dataset management
- Better for longer training runs

### Setup Steps:

#### 1. Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → GPU

#### 2. Install Dependencies

Add this to your notebook:

```python
# Install packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pillow matplotlib tqdm

# Clone repositories
!git clone https://github.com/LS4GAN/uvcgan4slats.git
%cd uvcgan4slats

!git clone https://github.com/LS4GAN/toytools
%cd toytools
!pip install -e .
%cd ..

!pip install -e .
```

#### 3. Upload Dataset

**Option A: Use Kaggle Datasets**
1. Create a new dataset on Kaggle
2. Upload your BRATS19 data as a zip file
3. Add dataset to notebook: Data → Add data → Your dataset

**Option B: Direct upload**
```python
# Upload via notebook (limited size)
from IPython.display import FileLink
# Or use Kaggle API
```

#### 4. Set Environment Variables

```python
import os
os.environ['UVCGAN_DATA'] = '/kaggle/working'
os.environ['UVCGAN_OUTDIR'] = '/kaggle/working/outputs'
```

#### 5. Run Training

Same as Colab - use `!python scripts/brats19/...`

### Kaggle Limitations:
- 30 hours/week GPU limit
- 20GB output limit
- Internet access disabled (need to add datasets)

---

## Option 3: Simplified Cloud Setup Script

Create a setup script that works on both platforms:

```python
# setup_cloud.py
import os
import subprocess

def setup_cloud_environment():
    """Setup UVCGAN environment for cloud GPU platforms"""
    
    # Detect platform
    if 'COLAB_GPU' in os.environ:
        platform = 'colab'
        data_dir = '/content'
        out_dir = '/content/outputs'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        platform = 'kaggle'
        data_dir = '/kaggle/working'
        out_dir = '/kaggle/working/outputs'
    else:
        platform = 'local'
        data_dir = os.environ.get('UVCGAN_DATA', './data')
        out_dir = os.environ.get('UVCGAN_OUTDIR', './outdir')
    
    # Set environment variables
    os.environ['UVCGAN_DATA'] = data_dir
    os.environ['UVCGAN_OUTDIR'] = out_dir
    
    print(f"Platform detected: {platform}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    
    # Install dependencies
    if platform != 'local':
        print("Installing dependencies...")
        subprocess.run(['pip', 'install', 'torch', 'torchvision', 'torchaudio', 
                       '--index-url', 'https://download.pytorch.org/whl/cu118'])
        subprocess.run(['pip', 'install', 'numpy', 'pillow', 'matplotlib', 'tqdm'])
    
    return platform, data_dir, out_dir

if __name__ == '__main__':
    setup_cloud_environment()
```

---

## Recommended Workflow for Cloud Training

### 1. Prepare Your Dataset Locally

```bash
# Compress your dataset
cd /path/to/your/data
zip -r brats19.zip brats19/
```

### 2. Upload to Cloud Storage

**For Colab:**
- Upload `brats19.zip` to Google Drive
- Or upload directly via Colab file upload

**For Kaggle:**
- Create a Kaggle dataset
- Upload `brats19.zip` as a dataset

### 3. Create Training Notebook

Create a notebook with these cells:

```python
# Cell 1: Setup
!git clone https://github.com/LS4GAN/uvcgan4slats.git
%cd uvcgan4slats

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pillow matplotlib tqdm

# Install toytools
!git clone https://github.com/LS4GAN/toytools
%cd toytools
!pip install -e .
%cd ..

# Install uvcgan4slats
!pip install -e .
```

```python
# Cell 2: Mount/Extract Dataset
# For Colab with Google Drive:
from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/brats19.zip -d /content/

# For Kaggle:
# Dataset should be in /kaggle/input/
# !unzip -q /kaggle/input/brats19/brats19.zip -d /kaggle/working/
```

```python
# Cell 3: Set Environment Variables
import os
os.environ['UVCGAN_DATA'] = '/content'  # Change for Kaggle: '/kaggle/working'
os.environ['UVCGAN_OUTDIR'] = '/content/outputs'  # Change for Kaggle: '/kaggle/working/outputs'
```

```python
# Cell 4: Run Pretraining
!python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 32
```

```python
# Cell 5: Run Main Training
!python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

```python
# Cell 6: Save Results (Colab)
!cp -r /content/outputs /content/drive/MyDrive/uvcgan_outputs
```

---

## Tips for Cloud Training

1. **Reduce Batch Size**: Cloud GPUs may have less memory, use `--batch_size 16` or `32` instead of `64`

2. **Save Checkpoints Frequently**: Cloud sessions can disconnect
   - Modify checkpoint frequency in scripts if needed
   - Save to cloud storage regularly

3. **Monitor Training**: Use Colab/Kaggle's built-in output or add logging

4. **Resume Training**: If session disconnects, you can resume from checkpoints:
   ```python
   # Modify scripts to load from checkpoint
   # Or use the transfer option in train_brats19.py
   ```

5. **Data Size**: 
   - Compress datasets before uploading
   - Consider using a subset for initial testing
   - Use data augmentation to maximize dataset utility

6. **Time Management**:
   - Colab: ~12 hours max per session
   - Kaggle: 30 hours/week total
   - Plan training epochs accordingly

---

## Alternative: CPU Training (Very Slow)

If GPU is not available, you can train on CPU but it will be **extremely slow**:

```python
# Force CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Use smaller batch size
!python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 4
```

**Not recommended** - training will take days/weeks instead of hours.

---

## Quick Start Template

Save this as a Colab/Kaggle notebook template:

```python
# ============================================================================
# UVCGAN Cloud GPU Setup
# ============================================================================

# 1. Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy pillow matplotlib tqdm gitpython

# 2. Clone repositories
!git clone https://github.com/LS4GAN/uvcgan4slats.git
%cd uvcgan4slats

!git clone https://github.com/LS4GAN/toytools
%cd toytools
!pip install -e .
%cd ..

!pip install -e .

# 3. Setup (modify paths for your platform)
import os
os.environ['UVCGAN_DATA'] = '/content'  # or '/kaggle/working'
os.environ['UVCGAN_OUTDIR'] = '/content/outputs'  # or '/kaggle/working/outputs'

# 4. Upload/extract your dataset here
# (Add your dataset upload/extraction code)

# 5. Run training
!python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 32
!python scripts/brats19/train_brats19.py --gen uvcgan --labmda-cycle 1.0 --lr-gen 1e-5 --lr-disc 5e-5
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 16` or `8`
- Reduce image size in scripts if possible
- Use gradient checkpointing if available

### Session Disconnected
- Save checkpoints frequently
- Use Colab Pro for longer sessions (paid)
- Save outputs to cloud storage regularly

### Import Errors
- Make sure you're in the correct directory
- Restart runtime after installing packages
- Check Python version compatibility

### Dataset Not Found
- Verify `UVCGAN_DATA` environment variable
- Check dataset path structure
- Ensure dataset is extracted/unzipped

