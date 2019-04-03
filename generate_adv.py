from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')

from PIL import Image
import numpy as np 
import os
import pdb

from perceptron.utils.image import letterbox_image, draw_letterbox
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.attacks.carlini_wagner import CarliniWagnerLinfAttack
from perceptron.utils.criteria.detection import TargetClassMiss, WeightedAP
from perceptron.defences.bit_depth import BitDepth

def main():
    video_name = 'cabc30fc-e7726578'
    input_benign_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "benign")
    output_benign_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv")
    output_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_bit_5")
    image_name_list = os.listdir(input_benign_dir)

    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))

    attack = CarliniWagnerLinfAttack(model, criterion=TargetClassMiss(2))
    squz_fn = BitDepth(5)

    for idx, image_name in enumerate(image_name_list):
        if idx % 10 == 0:
            print('idx : ', idx)

        temp_img_path_benign = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "benign", image_name)

        image_benign, shape = letterbox_image(
                data_format='channels_last', 
                fname=temp_img_path_benign
        )

        try:
            output_benign = model.predictions(image_benign)
            image_adv_benign = attack(image_benign, output_benign)
            image_adv_benign_pil = Image.fromarray((image_adv_benign * 255).astype(np.uint8))
            #image_adv_benign_pil.save(os.path.join(output_benign_dir, image_name))

            image_adv_squz = squz_fn(image_adv_benign)
            image_adv_squz_pil = Image.fromarray((image_adv_squz * 255).astype(np.uint8))
            #image_adv_squz_pil.save(os.path.join(output_squz_dir, image_name))
        except:
            print(image_name, ' generating error.')
            continue

def generate_squz_adv_from_adv():
    video_name = 'cabc30fc-e7726578'
    input_adv_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv")
    output_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_bit_5")
    image_name_list = os.listdir(input_adv_dir)

    squz_fn = BitDepth(5)

    for idx, image_name in enumerate(image_name_list):
        if idx % 10 == 0:
            print('idx : ', idx)

        temp_img_path_adv = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "adv", image_name)

        image_adv, shape = letterbox_image(
                data_format='channels_last', 
                fname=temp_img_path_adv
        )

        try:
            image_adv_squz = squz_fn(image_adv)
            image_adv_squz_pil = Image.fromarray((image_adv_squz * 255).astype(np.uint8))
            image_adv_squz_pil.save(os.path.join(output_squz_dir, image_name))
        except:
            print(image_name, ' generating error.')
            continue


if __name__ == "__main__":
    generate_squz_adv_from_adv()