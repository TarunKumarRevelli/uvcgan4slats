import argparse
import os

from uvcgan import ROOT_DATA, ROOT_OUTDIR, train
from uvcgan.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Pretrain BRATS19 BERT')
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'uvcgan', help_msg='generator type')
    add_batch_size_parser(parser, default = 64)
    return parser.parse_args()

GEN_PRESETS = {
    'resnet9' : {
        'model'      : 'resnet_9blocks',
        'model_args' : None,
    },
    'unet' : {
        'model'      : 'unet_256',
        'model_args' : None,
    },
    'resnet9-nonorm' : {
        'model'      : 'resnet_9blocks',
        'model_args' : {
            'norm' : 'none',
        },
    },
    'unet-nonorm' : {
        'model'      : 'unet_256',
        'model_args' : {
            'norm' : 'none',
        },
    },
    'uvcgan' : {
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
    },
}

# ============================================================================
# CONFIGURATION: Modify these values for your BRATS19 dataset
# ============================================================================
DATASET_PATH = 'brats19'  # Path relative to UVCGAN_DATA
DOMAIN_NAMES = ['t1', 't2']  # Change to your domain names (e.g., ['t1', 't2'] or ['flair', 't1ce'])
IMAGE_SHAPE = (1, 256, 256)  # (channels, height, width) - adjust to your image size
TARGET_SIZE = 256  # Target image size for resizing (height, width)

# Transforms for PNG images
# Images will be automatically converted to tensors (values 0-1) by ToTensor()
# Add 'resize' transform to ensure all images are the same size
# Note: If your PNGs are RGB (3 channels), change IMAGE_SHAPE to (3, 256, 256)
#       If your PNGs are grayscale (1 channel), keep IMAGE_SHAPE as (1, 256, 256)
TRANSFORMS_TRAIN = [
    {'name': 'resize', 'size': TARGET_SIZE},  # Resize to target size
    # {'name': 'random-flip-horizontal'},  # Optional: data augmentation
]

TRANSFORMS_TEST = [
    {'name': 'resize', 'size': TARGET_SIZE},  # Resize to target size
]
# ============================================================================

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'image-domain-hierarchy',
                    'domain' : domain,
                    'path'   : DATASET_PATH,
                },
                'shape'           : IMAGE_SHAPE,
                'transform_train' : TRANSFORMS_TRAIN if TRANSFORMS_TRAIN else None,
                'transform_test'  : TRANSFORMS_TEST if TRANSFORMS_TEST else None,
            } for domain in DOMAIN_NAMES
        ],
        'merge_type' : 'unpaired',
    },
    'epochs'        : 499,
    'discriminator' : None,
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-5 / 512,
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
        'eta_min' : cmdargs.batch_size * 5e-5 * 1e-5 / 512,
    },
    'loss'             : 'l2',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : 'pretrain-brats19-256',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'brats19'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 100,
}

train(args_dict)

