#!/usr/bin/env python
"""
Visualize UVCGAN translation results as side-by-side images.
Works with PNG images saved by translate_data.py or during evaluation.
"""

import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description='Visualize translation results as side-by-side images'
    )
    
    parser.add_argument(
        'result_dir',
        help='Directory containing translated images (e.g., fake_b, real_b, etc.)',
        type=str
    )
    
    parser.add_argument(
        'output_dir',
        help='Directory to save visualization plots',
        type=str
    )
    
    parser.add_argument(
        '--n-samples',
        dest='n_samples',
        type=int,
        default=10,
        help='Number of samples to visualize (default: 10)'
    )
    
    parser.add_argument(
        '--compare-with',
        dest='compare_with',
        type=str,
        default=None,
        help='Directory to compare with (e.g., real_b for fake_b)'
    )
    
    parser.add_argument(
        '--layout',
        choices=['horizontal', 'vertical', 'grid'],
        default='horizontal',
        help='Layout of comparison images (default: horizontal)'
    )
    
    parser.add_argument(
        '--cmap',
        default='gray',
        help='Colormap for grayscale images (default: gray)'
    )
    
    return parser.parse_args()

def load_image_file(path):
    """Load image from file (supports .png, .npz, .npy)"""
    if path.endswith('.npz'):
        data = np.load(path)
        # Get first array from npz
        img = data[data.files[0]]
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
    elif path.endswith('.npy'):
        img = np.load(path)
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
    else:
        # PNG or other image format
        img = np.array(Image.open(path).convert('L'))  # Convert to grayscale
    
    # Normalize to [0, 1] if needed
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    return img

def find_image_files(directory):
    """Find all image files in directory"""
    patterns = ['*.png', '*.npz', '*.npy']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    files.sort()
    return files

def plot_comparison(img1, img2, title1='Image 1', title2='Image 2', cmap='gray'):
    """Plot two images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img1, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(title1, fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(title2, fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def plot_grid(images, titles, cmap='gray', n_cols=2):
    """Plot multiple images in a grid"""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    cmdargs = parse_cmdargs()
    
    # Find image files
    result_files = find_image_files(cmdargs.result_dir)
    
    if not result_files:
        print(f"Error: No image files found in {cmdargs.result_dir}")
        return
    
    # Limit number of samples
    result_files = result_files[:cmdargs.n_samples]
    
    # Create output directory
    os.makedirs(cmdargs.output_dir, exist_ok=True)
    
    # Load comparison images if provided
    compare_files = None
    if cmdargs.compare_with and os.path.exists(cmdargs.compare_with):
        compare_files = find_image_files(cmdargs.compare_with)
        compare_files = compare_files[:cmdargs.n_samples]
    
    print(f"Visualizing {len(result_files)} samples...")
    
    # Visualize each sample
    for idx, result_file in enumerate(result_files):
        result_img = load_image_file(result_file)
        
        if compare_files and idx < len(compare_files):
            # Compare with another image
            compare_img = load_image_file(compare_files[idx])
            
            # Extract filenames for titles
            result_name = os.path.basename(result_file)
            compare_name = os.path.basename(compare_files[idx])
            
            if cmdargs.layout == 'horizontal':
                fig = plot_comparison(result_img, compare_img, result_name, compare_name, cmdargs.cmap)
            else:
                images = [result_img, compare_img]
                titles = [result_name, compare_name]
                fig = plot_grid(images, titles, cmdargs.cmap, n_cols=2)
        else:
            # Just show the result image
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(result_img, cmap=cmdargs.cmap, vmin=0, vmax=1)
            ax.set_title(os.path.basename(result_file), fontsize=12)
            ax.axis('off')
            plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(cmdargs.output_dir, f'sample_{idx:04d}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
    
    print(f"\n✓ Visualization complete! Saved {len(result_files)} images to {cmdargs.output_dir}")

if __name__ == '__main__':
    main()

