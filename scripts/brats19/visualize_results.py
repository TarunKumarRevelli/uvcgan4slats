#!/usr/bin/env python
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def parse_cmdargs():
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('result_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--compare-with', type=str, default=None)
    parser.add_argument('--layout', default='horizontal')
    parser.add_argument('--cmap', default='gray')
    # NEW ARGUMENTS FOR TITLES
    parser.add_argument('--title1', type=str, default=None, help='Title for first image')
    parser.add_argument('--title2', type=str, default=None, help='Title for second image')
    return parser.parse_args()

def load_image_file(path):
    if path.endswith('.npz'):
        data = np.load(path)
        img = data[data.files[0]] # Takes the first array (arr_0)
        if len(img.shape) == 3 and img.shape[0] == 1: img = img.squeeze(0)
    elif path.endswith('.npy'):
        img = np.load(path)
        if len(img.shape) == 3 and img.shape[0] == 1: img = img.squeeze(0)
    else:
        img = np.array(Image.open(path).convert('L'))
    
    if img.max() > 1.0: img = img.astype(np.float32) / 255.0
    return img

def plot_comparison(img1, img2, title1, title2, cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    return fig

def main():
    args = parse_cmdargs()
    
    patterns = ['*.png', '*.npz', '*.npy']
    result_files = []
    for p in patterns: result_files.extend(glob.glob(os.path.join(args.result_dir, p)))
    result_files.sort()
    result_files = result_files[:args.n_samples]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    compare_files = None
    if args.compare_with and os.path.exists(args.compare_with):
        compare_files = []
        for p in patterns: compare_files.extend(glob.glob(os.path.join(args.compare_with, p)))
        compare_files.sort()
        compare_files = compare_files[:args.n_samples]
    
    for idx, result_file in enumerate(result_files):
        img1 = load_image_file(result_file)
        # Use provided title or filename
        t1 = args.title1 if args.title1 else os.path.basename(result_file)
        
        if compare_files and idx < len(compare_files):
            img2 = load_image_file(compare_files[idx])
            t2 = args.title2 if args.title2 else os.path.basename(compare_files[idx])
            
            fig = plot_comparison(img1, img2, t1, t2, args.cmap)
        else:
            # Single image plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img1, cmap=args.cmap)
            ax.set_title(t1)
            ax.axis('off')

        out_path = os.path.join(args.output_dir, f'sample_{idx:03d}.png')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()