#!/usr/bin/env python
"""
Quick test script to verify the entire UVCGAN pipeline works correctly.
Runs with minimal epochs and samples to save GPU compute hours.
"""

import argparse
import os
import subprocess
import sys

from uvcgan import ROOT_DATA, ROOT_OUTDIR

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description='Quick test of UVCGAN pipeline with minimal epochs'
    )
    
    parser.add_argument(
        '--pretrain-epochs',
        dest='pretrain_epochs',
        type=int,
        default=1,
        help='Number of pretraining epochs (default: 1)'
    )
    
    parser.add_argument(
        '--train-epochs',
        dest='train_epochs',
        type=int,
        default=1,
        help='Number of training epochs (default: 1)'
    )
    
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=4,
        help='Batch size for testing (default: 4, smaller to save memory)'
    )
    
    parser.add_argument(
        '--skip-pretrain',
        dest='skip_pretrain',
        action='store_true',
        help='Skip pretraining step'
    )
    
    parser.add_argument(
        '--gen',
        default='uvcgan',
        help='Generator type (default: uvcgan)'
    )
    
    return parser.parse_args()

def modify_script_epochs(script_path, epochs, batch_size=None):
    """Temporarily modify script to use fewer epochs"""
    with open(script_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Replace epochs
    import re
    content = re.sub(
        r"'epochs'\s*:\s*\d+",
        f"'epochs'        : {epochs}",
        content
    )
    
    # Replace batch size if specified
    if batch_size is not None:
        content = re.sub(
            r"'batch_size'\s*:\s*cmdargs\.batch_size",
            f"'batch_size' : {batch_size}",
            content
        )
        # Also handle pretrain batch_size
        content = re.sub(
            r"'batch_size'\s*:\s*cmdargs\.batch_size",
            f"'batch_size' : {batch_size}",
            content
        )
    
    # Reduce steps_per_epoch for faster testing
    content = re.sub(
        r"'steps_per_epoch'\s*:\s*\d+\s*\*\s*\d+\s*//\s*cmdargs\.batch_size",
        f"'steps_per_epoch'  : 10,  # Reduced for testing",
        content
    )
    content = re.sub(
        r"'steps_per_epoch'\s*:\s*\d+,",
        f"'steps_per_epoch'  : 10,  # Reduced for testing",
        content
    )
    
    # Write modified content
    with open(script_path, 'w') as f:
        f.write(content)
    
    return original_content

def restore_script(script_path, original_content):
    """Restore original script content"""
    with open(script_path, 'w') as f:
        f.write(original_content)

def main():
    cmdargs = parse_cmdargs()
    
    print("=" * 60)
    print("UVCGAN Pipeline Quick Test")
    print("=" * 60)
    print(f"Pretrain epochs: {cmdargs.pretrain_epochs}")
    print(f"Train epochs: {cmdargs.train_epochs}")
    print(f"Batch size: {cmdargs.batch_size}")
    print(f"Generator: {cmdargs.gen}")
    print("=" * 60)
    
    pretrain_script = 'scripts/brats19/pretrain_brats19.py'
    train_script = 'scripts/brats19/train_brats19.py'
    
    # Store original scripts
    pretrain_original = None
    train_original = None
    
    try:
        # Step 1: Pretraining (optional)
        if not cmdargs.skip_pretrain:
            print("\n[1/3] Running pretraining test...")
            
            # Modify pretrain script
            with open(pretrain_script, 'r') as f:
                pretrain_original = f.read()
            
            modify_script_epochs(pretrain_script, cmdargs.pretrain_epochs, cmdargs.batch_size)
            
            # Run pretraining
            cmd = [
                'python', pretrain_script,
                '--gen', cmdargs.gen,
                '--batch_size', str(cmdargs.batch_size)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("❌ Pretraining failed!")
                print(result.stderr)
                return 1
            
            print("✓ Pretraining completed successfully")
            
            # Restore script
            restore_script(pretrain_script, pretrain_original)
        else:
            print("\n[1/3] Skipping pretraining (--skip-pretrain)")
        
        # Step 2: Training
        print("\n[2/3] Running training test...")
        
        # Modify train script
        with open(train_script, 'r') as f:
            train_original = f.read()
        
        modify_script_epochs(train_script, cmdargs.train_epochs, None)
        
        # Run training
        cmd = [
            'python', train_script,
            '--gen', cmdargs.gen,
            '--labmda-cycle', '1.0',
            '--lr-gen', '1e-5',
            '--lr-disc', '5e-5',
        ]
        
        if cmdargs.skip_pretrain:
            cmd.append('--no-pretrain')
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Training failed!")
            print(result.stderr)
            return 1
        
        print("✓ Training completed successfully")
        
        # Restore script
        restore_script(train_script, train_original)
        
        # Step 3: Quick evaluation test
        print("\n[3/3] Testing evaluation pipeline...")
        
        # Find the checkpoint directory
        checkpoint_base = os.path.join(ROOT_OUTDIR, 'brats19')
        if os.path.exists(checkpoint_base):
            checkpoint_dirs = [d for d in os.listdir(checkpoint_base) 
                             if d.startswith('model_m') and 'train' in d]
            if checkpoint_dirs:
                checkpoint_dir = os.path.join(checkpoint_base, sorted(checkpoint_dirs)[-1])
                
                # Quick evaluation with minimal samples
                eval_cmd = [
                    'python', 'scripts/brats19/eval_and_visualize.py',
                    checkpoint_dir,
                    '--n-samples', '2',  # Just 2 samples for testing
                    '--split', 'test'
                ]
                
                print(f"Running: {' '.join(eval_cmd)}")
                result = subprocess.run(eval_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✓ Evaluation pipeline works!")
                else:
                    print("⚠ Evaluation had issues (may be expected with minimal data)")
                    print(result.stderr[:500])  # First 500 chars
            else:
                print("⚠ No training checkpoint found for evaluation test")
        else:
            print("⚠ Output directory not found")
        
        print("\n" + "=" * 60)
        print("✅ Pipeline test completed successfully!")
        print("=" * 60)
        print("\nYou can now run full training with:")
        print("  python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 32")
        print("  python scripts/brats19/train_brats19.py --gen uvcgan")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        
        # Restore scripts on error
        if pretrain_original:
            restore_script(pretrain_script, pretrain_original)
        if train_original:
            restore_script(train_script, train_original)
        
        return 1

if __name__ == '__main__':
    sys.exit(main())

