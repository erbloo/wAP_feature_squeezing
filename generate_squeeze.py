from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/perceptron-benchmark')

import argparse
from PIL import Image
import os
import shutil
import numpy as np 
from tqdm import tqdm
import pdb

from perceptron.defences.bit_depth import BitDepth


def main(args):
    imgs_dir = args.imgs_dir
    input_dir = os.path.join(args.data_dir, imgs_dir, "benign")
    if args.source_type == 'ori':
        output_dir = os.path.join(args.data_dir, imgs_dir, args.squeeze_type)
    elif args.source_type == 'adv':
        output_dir = os.path.join(args.data_dir, imgs_dir, args.squeeze_type + '_adv')
    else:
        raise

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    image_name_list = os.listdir(input_dir)
    if args.squeeze_type[:4] == 'bit_':
        squzee_bit = int(args.squeeze_type.split('_')[-1])
        squz_fn = BitDepth(squzee_bit)
    else:
        raise ValueError('Invalid squeeze-type {0}'.format(args.squeeze_type))

    for _, image_name in enumerate(tqdm(image_name_list)):
        image_pil = Image.open(os.path.join(input_dir, image_name))
        image_np = np.array(image_pil) / 255.
        image_squz = squz_fn(image_np)
        image_squz_pil = Image.fromarray((image_squz * 255).astype(np.uint8))
        image_squz_pil.save(os.path.join(output_dir, image_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squeeze image generation.")
    parser.add_argument('source_type', type=str, choices=['ori', 'adv'])
    parser.add_argument('--data-dir', type=str, default='/home/yantao/workspace/datasets/wAP')
    parser.add_argument('--imgs-dir', type=str, default='bdd10k_test')
    parser.add_argument('--squeeze-type', type=str, default='bit_5')
    args = parser.parse_args()
    main(args)
