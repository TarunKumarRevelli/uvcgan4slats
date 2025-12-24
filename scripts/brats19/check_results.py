#!/usr/bin/env python
"""
Quick script to check and visualize results from a checkpoint.
"""

import argparse
import os
import glob
import subprocess
import sys

def find_latest_checkpoint(base_dir):
    """Find the latest checkpoint directory"""
    if not os.path.exists(base_dir):
        return None
    
    checkpoint_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('model_m'):
            checkpoint_dirs.append((item_path, os.path.getmtime(item_path)))
    
    if not checkpoint_dirs:
        return None
    
    # Sort by modification time, return latest
    checkpoint_dirs.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_dirs[0][0]

def main():
    parser = argparse.ArgumentParser(description='Check and visualize training results')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Path to checkpoint directory (default: find latest)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples to visualize (default: 10)'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'test'],
        default='test',
        help='Dataset split (default: test)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list checkpoints, do not visualize'
    )
    
    args = parser.parse_args()
    
    # Find checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        # Try to find from environment or default
        from uvcgan import ROOT_OUTDIR
        base_dir = os.path.join(ROOT_OUTDIR, 'brats19')
        checkpoint_dir = find_latest_checkpoint(base_dir)
        
        if checkpoint_dir is None:
            print("Error: No checkpoint directory found.")
            print(f"Looked in: {base_dir}")
            print("\nPlease specify checkpoint directory with --checkpoint-dir")
            return 1
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    print("=" * 60)
    print("UVCGAN Results Checker")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 60)
    
    # List checkpoint contents
    print("\n📁 Checkpoint Contents:")
    print(f"  {checkpoint_dir}")
    
    # Check for model files
    model_files = glob.glob(os.path.join(checkpoint_dir, 'net_*.pth'))
    if model_files:
        print(f"\n  ✓ Found {len(model_files)} model checkpoint(s)")
        for mf in sorted(model_files)[:5]:  # Show first 5
            print(f"    - {os.path.basename(mf)}")
        if len(model_files) > 5:
            print(f"    ... and {len(model_files) - 5} more")
    else:
        print("  ⚠ No model checkpoints found")
    
    # Check for history
    history_file = os.path.join(checkpoint_dir, 'history.csv')
    if os.path.exists(history_file):
        print(f"\n  ✓ Training history: {history_file}")
        # Show last few lines
        try:
            with open(history_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print(f"    Last epoch: {lines[-1].split(',')[0] if ',' in lines[-1] else 'N/A'}")
        except:
            pass
    else:
        print("  ⚠ No training history found")
    
    # Check for existing evaluations
    eval_base = os.path.join(checkpoint_dir, 'evals', args.split)
    if os.path.exists(eval_base):
        eval_dirs = [d for d in os.listdir(eval_base) if d.startswith('ndarrays_eval')]
        if eval_dirs:
            eval_dir = os.path.join(eval_base, sorted(eval_dirs)[-1])
            print(f"\n  ✓ Evaluation results found: {eval_dir}")
            
            # Check what's in there
            subdirs = ['fake_a', 'fake_b', 'real_a', 'real_b', 'reco_a', 'reco_b']
            for subdir in subdirs:
                subdir_path = os.path.join(eval_dir, subdir)
                if os.path.exists(subdir_path):
                    files = glob.glob(os.path.join(subdir_path, '*'))
                    print(f"    - {subdir}: {len(files)} files")
        else:
            print(f"\n  ⚠ No evaluation results found in {eval_base}")
    else:
        print(f"\n  ⚠ No evaluation directory found: {eval_base}")
    
    # Check for existing visualizations
    vis_dir = os.path.join(checkpoint_dir, 'visualizations')
    if os.path.exists(vis_dir):
        vis_subdirs = [d for d in os.listdir(vis_dir) if os.path.isdir(os.path.join(vis_dir, d))]
        if vis_subdirs:
            print(f"\n  ✓ Visualizations found: {vis_dir}")
            for subdir in vis_subdirs:
                subdir_path = os.path.join(vis_dir, subdir)
                images = glob.glob(os.path.join(subdir_path, '*.png'))
                print(f"    - {subdir}: {len(images)} images")
        else:
            print(f"\n  ⚠ No visualizations found in {vis_dir}")
    else:
        print(f"\n  ⚠ No visualizations directory found")
    
    if args.list_only:
        print("\n" + "=" * 60)
        return 0
    
    # Generate visualizations if not present
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    # Run eval_and_visualize
    eval_cmd = [
        'python', 'scripts/brats19/eval_and_visualize.py',
        checkpoint_dir,
        '--n-samples', str(args.n_samples),
        '--split', args.split
    ]
    
    print(f"\nRunning: {' '.join(eval_cmd)}\n")
    result = subprocess.run(eval_cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ Visualizations generated successfully!")
        print("=" * 60)
        
        # Show where results are
        vis_dir = os.path.join(checkpoint_dir, 'visualizations')
        if os.path.exists(vis_dir):
            print(f"\n📊 Visualization results saved to:")
            print(f"   {vis_dir}")
            
            for subdir in os.listdir(vis_dir):
                subdir_path = os.path.join(vis_dir, subdir)
                if os.path.isdir(subdir_path):
                    images = glob.glob(os.path.join(subdir_path, '*.png'))
                    print(f"\n   {subdir}/: {len(images)} images")
                    if images:
                        print(f"      Example: {os.path.basename(images[0])}")
        
        print("\n💡 To view images:")
        print(f"   - In Colab: Use Image() or display() on files in {vis_dir}")
        print(f"   - Locally: Open the PNG files in {vis_dir}")
        
    else:
        print("\n❌ Visualization generation had errors")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

