#!/usr/bin/env python
"""
Evaluate a checkpoint and visualize results.
This script loads a model checkpoint, generates translations, and creates visualizations.
"""

import argparse
import os
import subprocess
import sys

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description='Evaluate checkpoint and visualize results'
    )
    
    parser.add_argument(
        'checkpoint_path',
        help='Path to model checkpoint directory',
        type=str
    )
    
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch to evaluate (default: latest)'
    )
    
    parser.add_argument(
        '--n-samples',
        dest='n_samples',
        type=int,
        default=10,
        help='Number of samples to evaluate (default: 10)'
    )
    
    parser.add_argument(
        '--split',
        choices=['train', 'test'],
        default='test',
        help='Dataset split to evaluate (default: test)'
    )
    
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: checkpoint_dir/visualizations)'
    )
    
    return parser.parse_args()

def main():
    cmdargs = parse_cmdargs()
    
    # Determine checkpoint directory
    checkpoint_dir = cmdargs.checkpoint_path
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Set output directory
    if cmdargs.output_dir is None:
        output_dir = os.path.join(checkpoint_dir, 'visualizations')
    else:
        output_dir = cmdargs.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate translations using translate_data.py
    print("Step 1: Generating translations...")
    translate_cmd = [
        'python', 'scripts/translate_data.py',
        checkpoint_dir,
        '--n-eval', str(cmdargs.n_samples),
        '--split', cmdargs.split,
    ]
    
    if cmdargs.epoch is not None:
        translate_cmd.extend(['--epoch', str(cmdargs.epoch)])
    
    print(f"Running: {' '.join(translate_cmd)}")
    result = subprocess.run(translate_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running translate_data.py:")
        print(result.stderr)
        return
    
    print("✓ Translations generated")
    
    # Step 2: Find the evaluation directory
    # translate_data.py saves to evals/{split}/ndarrays_eval-{split}/
    eval_base = os.path.join(checkpoint_dir, 'evals', cmdargs.split)
    
    if not os.path.exists(eval_base):
        print(f"Error: Evaluation directory not found: {eval_base}")
        return
    
    # Find the most recent eval directory
    eval_dirs = [d for d in os.listdir(eval_base) if d.startswith('ndarrays_eval')]
    if not eval_dirs:
        print(f"Error: No evaluation results found in {eval_base}")
        return
    
    eval_dir = os.path.join(eval_base, sorted(eval_dirs)[-1])
    print(f"Found evaluation directory: {eval_dir}")
    
    # Step 3: Visualize results
    print("\nStep 2: Creating visualizations...")
    
    # Visualize fake_b vs real_b (translation quality)
    fake_b_dir = os.path.join(eval_dir, 'fake_b')
    real_b_dir = os.path.join(eval_dir, 'real_b')
    
    if os.path.exists(fake_b_dir) and os.path.exists(real_b_dir):
        vis_output = os.path.join(output_dir, 'fake_vs_real')
        os.makedirs(vis_output, exist_ok=True)
        
        vis_cmd = [
            'python', 'scripts/brats19/visualize_results.py',
            fake_b_dir,
            vis_output,
            '--n-samples', str(cmdargs.n_samples),
            '--compare-with', real_b_dir,
            '--layout', 'horizontal',
        ]
        
        print(f"Running: {' '.join(vis_cmd)}")
        result = subprocess.run(vis_cmd)
        
        if result.returncode == 0:
            print(f"✓ Visualizations saved to {vis_output}")
        else:
            print("Error creating visualizations")
    
    # Visualize fake_a vs real_a
    fake_a_dir = os.path.join(eval_dir, 'fake_a')
    real_a_dir = os.path.join(eval_dir, 'real_a')
    
    if os.path.exists(fake_a_dir) and os.path.exists(real_a_dir):
        vis_output = os.path.join(output_dir, 'fake_a_vs_real_a')
        os.makedirs(vis_output, exist_ok=True)
        
        vis_cmd = [
            'python', 'scripts/brats19/visualize_results.py',
            fake_a_dir,
            vis_output,
            '--n-samples', str(cmdargs.n_samples),
            '--compare-with', real_a_dir,
            '--layout', 'horizontal',
        ]
        
        print(f"Running: {' '.join(vis_cmd)}")
        subprocess.run(vis_cmd)
        print(f"✓ Visualizations saved to {vis_output}")
    
    # Visualize cycle consistency (reco_a vs real_a)
    reco_a_dir = os.path.join(eval_dir, 'reco_a')
    if os.path.exists(reco_a_dir) and os.path.exists(real_a_dir):
        vis_output = os.path.join(output_dir, 'cycle_consistency_a')
        os.makedirs(vis_output, exist_ok=True)
        
        vis_cmd = [
            'python', 'scripts/brats19/visualize_results.py',
            reco_a_dir,
            vis_output,
            '--n-samples', str(cmdargs.n_samples),
            '--compare-with', real_a_dir,
            '--layout', 'horizontal',
        ]
        
        print(f"Running: {' '.join(vis_cmd)}")
        subprocess.run(vis_cmd)
        print(f"✓ Cycle consistency visualizations saved to {vis_output}")
    
    print(f"\n✅ Complete! All visualizations saved to {output_dir}")

if __name__ == '__main__':
    main()

