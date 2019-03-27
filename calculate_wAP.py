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
    dir_path = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "benign")
    image_name_list = os.listdir(dir_path)

    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))

    squz_fn = BitDepth(6)
    weighted_ap = WeightedAP(416, 416, 0.0001)

    scores_benign = []
    scores_adv = []
    with open(video_name + '.csv', 'w') as out_file:
        out_file.write('{0},{1}\n'.format('benign', 'adv'))

    for idx, image_name in enumerate(image_name_list):
        if idx % 10 == 0:
            print('idx : ', idx)
        temp_img_path = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "benign", image_name)
        image, shape = letterbox_image(
                data_format='channels_last', 
                fname=temp_img_path
        ) 

        output_benign = model.predictions(image)
        #print(output_benign)
        #draw = draw_letterbox(image, output_benign, shape[:2], model.class_names())
        #draw.save('benign.png')

        image_squz = squz_fn(image)
        output_benign_squz = model.predictions(image_squz)
        #print(output_benign_squz)
        #draw = draw_letterbox(image_squz, output_benign_squz, shape[:2], model.class_names())
        #draw.save('benign_squz.png')

        wAP_score_benign = weighted_ap.distance_score(output_benign, output_benign_squz)
        scores_benign.append(wAP_score_benign)

        attack = CarliniWagnerLinfAttack(model, criterion=TargetClassMiss(2))
        image_adv = attack(image, output_benign)
        output_adv = model.predictions(image_adv)
        #print(output_adv)
        #draw = draw_letterbox(image_adv, output_adv, shape[:2], model.class_names())
        #draw.save('adv.png')

        image_adv_squz = squz_fn(image_adv)
        output_adv_squz = model.predictions(image_adv_squz)
        #print(output_adv_squz)
        #draw = draw_letterbox(image_adv_squz, output_adv_squz, shape[:2], model.class_names())
        #draw.save('adv_squz.png')

        wAP_score_adv = weighted_ap.distance_score(output_adv, output_adv_squz)
        scores_adv.append(wAP_score_adv)

        print('wAP benign : ', wAP_score_benign)
        print('wAP adv : ', wAP_score_adv)

        with open(video_name + '.csv', 'a') as out_file:
            out_file.write('{0},{1}\n'.format(str(wAP_score_benign), str(wAP_score_adv)))

    





if __name__ == "__main__":
    main()