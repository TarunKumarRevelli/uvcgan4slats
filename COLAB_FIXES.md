# Colab Fixes and Troubleshooting

## Config Collision Error

If you get a "Config collision detected" error, it means you're trying to resume training with a different configuration than what was saved. This happens when you:
- Changed transforms (e.g., added grayscale)
- Changed other config parameters
- Want to start fresh

### Solution 1: Delete Old Checkpoint (Recommended)

Add this cell in Colab **before** running pretraining:

```python
# Delete old checkpoint directory to start fresh
import shutil
import os

checkpoint_dir = '/content/outputs/brats19/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256'

if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
    print(f"✓ Deleted old checkpoint: {checkpoint_dir}")
else:
    print("No old checkpoint found, starting fresh")
```

### Solution 2: Use a New Label

Modify the label in the script to create a new output directory. In Colab, you can override it:

```python
# After cloning the repo, modify the label
import re

with open('scripts/brats19/pretrain_brats19.py', 'r') as f:
    content = f.read()

# Change label to create new directory
content = content.replace(
    "'label'      : 'pretrain-brats19-256',",
    "'label'      : 'pretrain-brats19-256-v2',  # New label for new config"
)

with open('scripts/brats19/pretrain_brats19.py', 'w') as f:
    f.write(content)

print("✓ Updated label to create new checkpoint directory")
```

### Solution 3: Quick Fix Cell for Colab

Add this cell that handles everything:

```python
# Fix config collision - delete old checkpoint
import shutil
import os

output_base = '/content/outputs/brats19'
old_checkpoint = os.path.join(output_base, 'model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256')

if os.path.exists(old_checkpoint):
    print(f"Deleting old checkpoint: {old_checkpoint}")
    shutil.rmtree(old_checkpoint)
    print("✓ Old checkpoint deleted - ready for fresh training")
else:
    print("No old checkpoint found - ready to start training")
```

## Complete Colab Setup Cell

Here's a complete setup cell that handles all fixes:

```python
# ============================================================================
# Complete Setup Cell for Colab
# ============================================================================

import os
import shutil
import re

# 1. Fix scheduler verbose issue
print("1. Fixing scheduler...")
with open('uvcgan/base/schedulers.py', 'r') as f:
    content = f.read()

# Fix verbose parameter
if "kwargs['verbose'] = True" in content and "if name != 'CosineAnnealingWarmRestarts':" not in content:
    content = content.replace(
        "    kwargs['verbose'] = True\n\n    if name == 'linear':",
        "    # Add verbose only to schedulers that support it\n    if name != 'CosineAnnealingWarmRestarts':\n        kwargs['verbose'] = True\n\n    if name == 'linear':"
    )
    content = content.replace(
        "    if name == 'CosineAnnealingWarmRestarts':\n        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)",
        "    if name == 'CosineAnnealingWarmRestarts':\n        # Remove verbose if present\n        kwargs.pop('verbose', None)\n        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)"
    )
    with open('uvcgan/base/schedulers.py', 'w') as f:
        f.write(content)
    print("   ✓ Scheduler fixed")

# 2. Add grayscale transform
print("2. Adding grayscale transform...")
with open('uvcgan/data/transforms.py', 'r') as f:
    content = f.read()

if "'grayscale'" not in content:
    content = content.replace(
        "'resize'                 : transforms.Resize,",
        "'resize'                 : transforms.Resize,\n    'grayscale'              : transforms.Grayscale,"
    )
    content = content.replace(
        "'Resize'                 : transforms.Resize,",
        "'Resize'                 : transforms.Resize,\n    'Grayscale'              : transforms.Grayscale,"
    )
    with open('uvcgan/data/transforms.py', 'w') as f:
        f.write(content)
    print("   ✓ Grayscale transform added")

# 3. Update scripts with grayscale
print("3. Updating training scripts...")
for script in ['scripts/brats19/pretrain_brats19.py', 'scripts/brats19/train_brats19.py']:
    with open(script, 'r') as f:
        content = f.read()
    
    # Add grayscale to transforms if not present
    if "{'name': 'grayscale'}" not in content:
        content = re.sub(
            r"(\{'name': 'resize', 'size': TARGET_SIZE\}),?\s*#",
            r"\1,\n    {'name': 'grayscale'},  #",
            content
        )
        content = re.sub(
            r"(\{'name': 'resize', 'size': TARGET_SIZE\})\s*\n\]",
            r"\1,\n    {'name': 'grayscale'}\n]",
            content
        )
        
        with open(script, 'w') as f:
            f.write(content)
    print(f"   ✓ Updated {script}")

# 4. Delete old checkpoint to avoid config collision
print("4. Cleaning old checkpoints...")
checkpoint_dir = '/content/outputs/brats19/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256'
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
    print(f"   ✓ Deleted old checkpoint")
else:
    print("   ✓ No old checkpoint found")

print("\n✅ All fixes applied! Ready to train.")
```

## Alternative: Change Label in Scripts

If you want to keep the old checkpoint and start a new one, modify the label:

```python
# Change label in pretrain script
with open('scripts/brats19/pretrain_brats19.py', 'r') as f:
    content = f.read()

content = content.replace(
    "'label'      : 'pretrain-brats19-256',",
    "'label'      : 'pretrain-brats19-256-grayscale',  # New config with grayscale"
)

with open('scripts/brats19/pretrain_brats19.py', 'w') as f:
    f.write(content)

# Also update train script to match
with open('scripts/brats19/train_brats19.py', 'r') as f:
    content = f.read()

# Update transfer path
content = content.replace(
    "'brats19/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256'",
    "'brats19/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256-grayscale'"
)

with open('scripts/brats19/train_brats19.py', 'w') as f:
    f.write(content)

print("✓ Updated labels - will create new checkpoint directory")
```

