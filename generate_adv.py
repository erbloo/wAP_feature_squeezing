from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/perceptron-benchmark')

from PIL import Image
import numpy as np 
import os
import shutil
import pdb
from tqdm import tqdm

from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.utils.image import load_image
from perceptron.benchmarks.carlini_wagner import CarliniWagnerLinfMetric
from perceptron.utils.criteria.detection import TargetClassMiss

def main():
    video_name = 'bdd10k_test'
    input_dir = os.path.join("/home/yantao/workspace/datasets/wAP", video_name, "benign")
    output_dir = os.path.join("/home/yantao/workspace/datasets/wAP", video_name, "adv")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    image_name_list = os.listdir(input_dir)

    kmodel = YOLOv3()
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))
    attack = CarliniWagnerLinfMetric(model, criterion=TargetClassMiss(2))

    for idx, image_name in enumerate(tqdm(image_name_list)):

        temp_img_path_benign = os.path.join(input_dir, image_name)

        image_benign = load_image(
                shape=(416, 416), bounds=(0, 1),
                fname=temp_img_path_benign,
                absolute_path=True
        )

        image_adv_benign = attack(image_benign, binary_search_steps=1, unpack=True)
        image_adv_benign_pil = Image.fromarray((image_adv_benign * 255).astype(np.uint8))
        image_adv_benign_pil.save(os.path.join(output_dir, image_name))


if __name__ == "__main__":
    main()