#!/usr/bin/env python
import argparse
import os
import subprocess
import glob

def parse_cmdargs():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint and visualize results')
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint directory')
    parser.add_argument('--epoch', type=int, default=None, help='Specific epoch (default: latest/final)')
    parser.add_argument('--n-samples', dest='n_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--split', choices=['train', 'test'], default='test', help='Dataset split')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='Output dir')
    return parser.parse_args()

def main():
    cmdargs = parse_cmdargs()
    
    checkpoint_dir = cmdargs.checkpoint_path
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Set output directory
    output_dir = cmdargs.output_dir if cmdargs.output_dir else os.path.join(checkpoint_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Step 1: Generate translations ---
    print(f"Step 1: Generating translations for split '{cmdargs.split}'...")
    translate_cmd = [
        'python', 'scripts/translate_data.py',
        checkpoint_dir,
        '-n', str(cmdargs.n_samples),
        '--split', cmdargs.split,
    ]
    if cmdargs.epoch is not None:
        translate_cmd.extend(['--epoch', str(cmdargs.epoch)])
    
    print(f"Running: {' '.join(translate_cmd)}")
    result = subprocess.run(translate_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running translate_data.py:\n{result.stderr}")
        return
    print("✓ Translations generated successfully.")
    
    # --- Step 2: Locate Evaluation Directory ---
    # We search for any folder starting with 'ndarrays' inside evals/{epoch} or evals/
    epoch_name = str(cmdargs.epoch) if cmdargs.epoch is not None else 'final'
    
    possible_roots = [
        os.path.join(checkpoint_dir, 'evals', epoch_name), # Check evals/final/
        os.path.join(checkpoint_dir, 'evals')              # Check evals/
    ]
    
    eval_base = None
    for root in possible_roots:
        if os.path.exists(root):
            # Find any folder starting with 'ndarrays'
            candidates = [d for d in os.listdir(root) if d.startswith('ndarrays')]
            if candidates:
                # Pick the most recent one if multiple exist
                eval_base = os.path.join(root, sorted(candidates)[-1])
                break
    
    if not eval_base:
        print(f"❌ Error: Could not find any 'ndarrays' folder in {checkpoint_dir}/evals")
        return

    print(f"📂 Found evaluation data at: {eval_base}")
    
    # --- Step 3: Visualize results ---
    print("\nStep 2: Creating visualizations...")
    
    def run_vis(source_sub, output_sub, title1, compare_sub=None, title2=None):
        src_path = os.path.join(eval_base, source_sub)
        out_path = os.path.join(output_dir, output_sub)
        
        if not os.path.exists(src_path):
            print(f"⚠️ Warning: Source folder {src_path} not found. Skipping {output_sub}.")
            return

        os.makedirs(out_path, exist_ok=True)
        
        cmd = [
            'python', 'scripts/brats19/visualize_results.py',
            src_path, out_path,
            '--n-samples', str(cmdargs.n_samples),
            '--title1', title1  # Custom title support
        ]
        
        if compare_sub:
            compare_path = os.path.join(eval_base, compare_sub)
            if os.path.exists(compare_path):
                cmd.extend(['--compare-with', compare_path])
                cmd.extend(['--title2', title2])
        
        print(f"Generating {output_sub}...")
        subprocess.run(cmd)

    # 1. Fake B (Prediction) vs Real B (Target)
    run_vis('fake_b', 'fake_vs_real', 'Generated (Fake B)', 'real_b', 'Ground Truth (Real B)')
    
    # 2. Fake A (Prediction) vs Real A (Target) - if doing B->A
    run_vis('fake_a', 'fake_a_vs_real_a', 'Generated (Fake A)', 'real_a', 'Ground Truth (Real A)')
    
    print(f"\n✅ Complete! All visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()