import argparse
import os

from uvcgan import ROOT_OUTDIR, train
from uvcgan.utils.parsers import add_preset_name_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train BRATS19 translation'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan', help_msg='generator type')

    parser.add_argument('--labmda-cycle',
                        dest = 'lambda_cyc',
                        type = float,
                        default = 1.0,
                        help = 'magnitude of the cycle-consisntecy loss (default = 1.0)')

    parser.add_argument('--lr-disc',
                        dest = 'lr_disc',
                        type = float,
                        default = 5e-5,
                        help = 'learning rate of the discriminator (default = 5e-5)')

    parser.add_argument('--lr-gen',
                        dest = 'lr_gen',
                        type = float,
                        default = 1e-5,
                        help = 'learning rate of the generator (default = 1e-5)')

    parser.add_argument('--gp-constant',
                        dest = 'constant_gp',
                        type = float,
                        default = 10.,
                        help = ('the constant gamma for gradient penalty (default = 10). '
                                'See the UVCGAN paper (https://arxiv.org/pdf/2203.02557.pdf) '
                                'section 3.3 for more detail.'))

    parser.add_argument('--gp-lambda',
                        dest = 'lambda_gp',
                        type = float,
                        default = 1.,
                        help = ('the coefficient of the gradient penalty (default = 1). '
                                'See the UVCGAN paper (https://arxiv.org/pdf/2203.02557.pdf) '
                                'section 3.3 for more detail.'))

    parser.add_argument('--no-pretrain',
                        dest = 'no_pretrain',
                        action = 'store_true',
                        help = 'Skip pretrained model transfer (train from scratch)')

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
    'batch_size' : 1,
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
    'epochs'        : 500,
    'discriminator' : {
        'model' : 'basic',
        'model_args' : None,
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_disc,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_gen,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'cyclegan',
    'model_args' : {
        'lambda_a'   : cmdargs.lambda_cyc,
        'lambda_b'   : cmdargs.lambda_cyc,
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
        'constant'  : cmdargs.constant_gp,
        'lambda_gp' : cmdargs.lambda_gp / (cmdargs.constant_gp ** 2),
    },
    'steps_per_epoch'  : 2000,
    'transfer' : None if cmdargs.no_pretrain else {
        'base_model'   : (
            'brats19/model_m(autoencoder)_d(None)_g(vit-unet)_pretrain-brats19-256'
        ),
        'transfer_map' : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
    },
# args
    'label' : (
        f'train-{cmdargs.gen}'
        f'-({cmdargs.lambda_cyc}:{cmdargs.lr_gen}:{cmdargs.lr_disc})'
        '_brats19-256'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'brats19'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
}

train(args_dict)

