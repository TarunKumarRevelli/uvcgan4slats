#!/usr/bin/env python
"""
Simpler quick test - just modify epochs directly via command line.
This is safer than modifying script files.
"""

import argparse
import os
import sys

# Import the training functions
from uvcgan import ROOT_OUTDIR, train

def create_test_config(pretrain_epochs=1, train_epochs=1, batch_size=4, skip_pretrain=False):
    """Create minimal test configurations"""
    
    # Pretrain config
    pretrain_config = {
        'batch_size' : batch_size,
        'data' : {
            'datasets' : [
                {
                    'dataset' : {
                        'name'   : 'image-domain-hierarchy',
                        'domain' : domain,
                        'path'   : 'brats19',
                    },
                    'shape'           : (1, 256, 256),
                    'transform_train' : [
                        {'name': 'resize', 'size': 256},
                        {'name': 'grayscale'},
                    ],
                    'transform_test'  : [
                        {'name': 'resize', 'size': 256},
                        {'name': 'grayscale'},
                    ],
                } for domain in ['t1', 't2']
            ],
            'merge_type' : 'unpaired',
        },
        'epochs'        : pretrain_epochs,
        'discriminator' : None,
        'generator' : {
            'model' : 'vit-unet',
            'model_args' : {
                'features'           : 384,
                'n_heads'            : 6,
                'n_blocks'           : 12,
                'ffn_features'       : 1536,
                'embed_features'     : 384,
                'activ'              : 'gelu',
                'norm'               : 'layer',
                'unet_features_list' : [48, 96, 192, 384],
                'unet_activ'         : 'leakyrelu',
                'unet_norm'          : None,
                'unet_downsample'    : 'conv',
                'unet_upsample'      : 'upsample-conv',
                'rezero'             : True,
                'activ_output'       : None,
            },
            'optimizer'  : {
                'name'  : 'AdamW',
                'lr'    : batch_size * 5e-5 / 512,
                'betas' : (0.9, 0.99),
                'weight_decay' : 0.05,
            },
            'weight_init' : {
                'name'      : 'normal',
                'init_gain' : 0.02,
            }
        },
        'model'      : 'autoencoder',
        'model_args' : {
            'joint'   : True,
            'background_penalty' : {
                'epochs_warmup' : 25,
                'epochs_anneal' : 75,
            },
            'masking' : {
                'name'       : 'image-patch-random',
                'patch_size' : (32, 32),
                'fraction'   : 0.4,
            },
        },
        'scheduler' : {
            'name'    : 'CosineAnnealingWarmRestarts',
            'T_0'     : 100,
            'T_mult'  : 1,
            'eta_min' : batch_size * 5e-5 * 1e-5 / 512,
        },
        'loss'             : 'l2',
        'gradient_penalty' : None,
        'steps_per_epoch'  : 10,  # Minimal for testing
        'label'      : 'test-pretrain-brats19',
        'outdir'     : os.path.join(ROOT_OUTDIR, 'brats19'),
        'log_level'  : 'INFO',
        'checkpoint' : 1,
    }
    
    # Train config
    train_config = {
        'batch_size' : 1,
        'data' : {
            'datasets' : [
                {
                    'dataset' : {
                        'name'   : 'image-domain-hierarchy',
                        'domain' : domain,
                        'path'   : 'brats19',
                    },
                    'shape'           : (1, 256, 256),
                    'transform_train' : [
                        {'name': 'resize', 'size': 256},
                        {'name': 'grayscale'},
                    ],
                    'transform_test'  : [
                        {'name': 'resize', 'size': 256},
                        {'name': 'grayscale'},
                    ],
                } for domain in ['t1', 't2']
            ],
            'merge_type' : 'unpaired',
        },
        'epochs'        : train_epochs,
        'discriminator' : {
            'model' : 'basic',
            'model_args' : None,
            'optimizer'  : {
                'name'  : 'Adam',
                'lr'    : 5e-5,
                'betas' : (0.5, 0.99),
            },
            'weight_init' : {
                'name'      : 'normal',
                'init_gain' : 0.02,
            },
        },
        'generator' : {
            'model' : 'vit-unet',
            'model_args' : {
                'features'           : 384,
                'n_heads'            : 6,
                'n_blocks'           : 12,
                'ffn_features'       : 1536,
                'embed_features'     : 384,
                'activ'              : 'gelu',
                'norm'               : 'layer',
                'unet_features_list' : [48, 96, 192, 384],
                'unet_activ'         : 'leakyrelu',
                'unet_norm'          : None,
                'unet_downsample'    : 'conv',
                'unet_upsample'      : 'upsample-conv',
                'rezero'             : True,
                'activ_output'       : None,
            },
            'optimizer'  : {
                'name'  : 'Adam',
                'lr'    : 1e-5,
                'betas' : (0.5, 0.99),
            },
            'weight_init' : {
                'name'      : 'normal',
                'init_gain' : 0.02,
            },
        },
        'model' : 'cyclegan',
        'model_args' : {
            'lambda_a'   : 1.0,
            'lambda_b'   : 1.0,
            'lambda_idt' : 0.5,
            'pool_size'  : 50,
        },
        'scheduler' : {
            'name'          : 'linear',
            'epochs_warmup' : 250,
            'epochs_anneal' : 250,
        },
        'loss' : 'lsgan',
        'gradient_penalty' : {
            'constant'  : 10.0,
            'lambda_gp' : 1.0 / (10.0 ** 2),
        },
        'steps_per_epoch'  : 10,  # Minimal for testing
        'transfer' : None if skip_pretrain else {
            'base_model'   : 'brats19/model_m(autoencoder)_d(None)_g(vit-unet)_test-pretrain-brats19',
            'transfer_map' : {
                'gen_ab' : 'encoder',
                'gen_ba' : 'encoder',
            },
            'strict'        : True,
            'allow_partial' : False,
        },
        'label' : 'test-train-brats19',
        'outdir' : os.path.join(ROOT_OUTDIR, 'brats19'),
        'log_level'  : 'INFO',
        'checkpoint' : 1,
    }
    
    return pretrain_config, train_config

def main():
    parser = argparse.ArgumentParser(description='Quick test of UVCGAN pipeline')
    parser.add_argument('--pretrain-epochs', type=int, default=1, help='Pretrain epochs (default: 1)')
    parser.add_argument('--train-epochs', type=int, default=1, help='Train epochs (default: 1)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip pretraining')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UVCGAN Quick Test")
    print("=" * 60)
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Train epochs: {args.train_epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    pretrain_config, train_config = create_test_config(
        args.pretrain_epochs, args.train_epochs, args.batch_size, args.skip_pretrain
    )
    
    try:
        # Step 1: Pretraining
        if not args.skip_pretrain:
            print("\n[1/2] Testing pretraining...")
            train(pretrain_config)
            print("✓ Pretraining test passed")
        
        # Step 2: Training
        print("\n[2/2] Testing training...")
        train(train_config)
        print("✓ Training test passed")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Pipeline is working correctly.")
        print("=" * 60)
        print("\nYou can now run full training:")
        print("  python scripts/brats19/pretrain_brats19.py --gen uvcgan --batch_size 32")
        print("  python scripts/brats19/train_brats19.py --gen uvcgan")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

