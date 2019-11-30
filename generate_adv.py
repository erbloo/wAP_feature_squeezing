from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/perceptron-benchmark')

from PIL import Image
import argparse
import numpy as np 
import os
import shutil
import pdb
from tqdm import tqdm

from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.utils.image import load_image
from perceptron.benchmarks.carlini_wagner import CarliniWagnerLinfMetric
from perceptron.benchmarks.additive_noise import AdditiveGaussianNoiseMetric
from perceptron.benchmarks.brightness import BrightnessMetric
from perceptron.utils.criteria.detection import TargetClassMiss, TargetClassNumberChange

DEBUG = False

def main(args):
    imgs_dir = args.imgs_dir
    input_dir = os.path.join(args.data_dir, imgs_dir, "benign")
    if not DEBUG:
        output_dir = os.path.join(args.data_dir, imgs_dir, "adv")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    image_name_list = os.listdir(input_dir)

    kmodel = YOLOv3()
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))
    if args.attack_mtd == 'cw_targetclsmiss':
        attack = CarliniWagnerLinfMetric(model, criterion=TargetClassMiss(2))
    elif args.attack_mtd == 'noise_numberchange':
        attack = AdditiveGaussianNoiseMetric(model, criterion=TargetClassNumberChange(-1))
    elif args.attack_mtd == 'bright_numberchange':
        attack = BrightnessMetric(model, criterion=TargetClassNumberChange(-1))
    else:
        raise ValueError('Invalid attack method {0}'.format( args.attack_mtd))

    for _, image_name in enumerate(tqdm(image_name_list)):

        temp_img_path_benign = os.path.join(input_dir, image_name)
        image_benign = load_image(
                shape=(416, 416), bounds=(0, 1),
                fname=temp_img_path_benign,
                absolute_path=True
        )

        try:
            annotation_ori = model.predictions(image_benign)
            if args.attack_mtd == 'cw_targetclsmiss':
                image_adv_benign = attack(image_benign, binary_search_steps=1, unpack=True)
            else
                image_adv_benign = attack(image_benign, annotation_ori, epsilons=1000, unpack=True)
        except:
            print('Attack failed.')
            continue

        image_adv_benign_pil = Image.fromarray((image_adv_benign * 255).astype(np.uint8))
        if not DEBUG:
            image_adv_benign_pil.save(os.path.join(output_dir, image_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squeeze image generation.")
    parser.add_argument('--data-dir', type=str, default='/home/yantao/workspace/datasets/wAP')
    parser.add_argument('--imgs-dir', type=str, default='bdd10k_test')
    parser.add_argument('--attack-mtd', type=str, default='cw_targetclsmiss')
    args = parser.parse_args()
    main(args)