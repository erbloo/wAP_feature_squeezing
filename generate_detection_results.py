from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')

from PIL import Image
import numpy as np 
import os
import pdb
import pickle

from perceptron.utils.image import letterbox_image, draw_letterbox
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.attacks.carlini_wagner import CarliniWagnerLinfAttack
from perceptron.utils.criteria.detection import TargetClassMiss, WeightedAP
from perceptron.defences.bit_depth import BitDepth

def main():
    video_name = 'cabc30fc-e7726578'
    input_benign_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "benign")
    input_adv_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv")
    input_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "bit_5")
    input_adv_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_bit_5")

    output_benign_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "benign_det_results")
    output_adv_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_det_results")
    output_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "bit_5_det_results")
    output_adv_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_bit_5_det_results")

    image_name_list = os.listdir(input_benign_dir)

    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))

    for idx, image_name in enumerate(image_name_list):
        if idx % 10 == 0:
            print('idx : ', idx)
        image_name_noext = os.path.splitext(image_name)[0]

        temp_img_path_benign = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "benign", image_name)
        temp_img_path_squz = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "bit_5", image_name)
        temp_img_path_adv = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "adv", image_name)
        temp_img_path_adv_squz = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "adv_bit_5", image_name)
        try:
            image_benign, shape = letterbox_image(
                    data_format='channels_last', 
                    fname=temp_img_path_benign
            )
            image_benign_squz, _ = letterbox_image(
                    data_format='channels_last', 
                    fname=temp_img_path_squz
            )
            image_adv, _ = letterbox_image(
                    data_format='channels_last', 
                    fname=temp_img_path_adv
            )
            image_adv_squz, _ = letterbox_image(
                    data_format='channels_last', 
                    fname=temp_img_path_adv_squz
            )
        except:
            print('loading images error.')
            continue
        
        try:
            output_benign = model.predictions(image_benign)
            output_adv = model.predictions(image_adv)
            output_benign_squz = model.predictions(image_benign_squz)
            output_adv_squz = model.predictions(image_adv_squz)
            '''
            with open(os.path.join(output_benign_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_benign, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output_adv_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_adv, f, protocol=pickle.HIGHEST_PROTOCOL)
            '''
            with open(os.path.join(output_squz_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_benign_squz, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(output_adv_squz_dir, image_name_noext + '.pkl'), 'wb') as f:
                pickle.dump(output_adv_squz, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        except:
            print(image_name, ' generating error.')
            continue
        


if __name__ == "__main__":
    main()