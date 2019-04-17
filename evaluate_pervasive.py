from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')

from PIL import Image
import numpy as np 
import os
import pdb
from collections import defaultdict
import pickle
import shutil

from perceptron.utils.image import letterbox_image, draw_letterbox
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.models.detection.keras_retina_resnet50 import KerasResNet50RetinaNetModel
from perceptron.zoo.retinanet_resnet_50.retina_resnet50 import retina_resnet50
from perceptron.benchmarks.carlini_wagner import CarliniWagnerLinfMetric
from perceptron.utils.criteria.detection import TargetClassMiss, WeightedAP
from perceptron.defences.bit_depth import BitDepth
from calculate_wAP import cal_mAP

def generate_adv(benign_dir, adv_dir):
    img_name_list = os.listdir(benign_dir)
    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))
    attack = CarliniWagnerLinfMetric(model, criterion=TargetClassMiss(2))
    for idx, image_name in enumerate(img_name_list):
        if idx % 10 == 0:
            print(idx)
        
        if os.path.exists(os.path.join(adv_dir, image_name)):
            continue
        
        temp_img_path_benign = os.path.join("../../../../../../../../datasets/bdd_parts/test_pervasive/benign", image_name)

        image_benign, shape = letterbox_image(
                data_format='channels_last', 
                fname=temp_img_path_benign
        )

        try:
            output_benign = model.predictions(image_benign)
        except:
            os.remove(os.path.join(benign_dir, image_name))
            continue

        if 2 not in output_benign['classes']:
            os.remove(os.path.join(benign_dir, image_name))
            continue

        try:
            image_adv_benign = attack(image_benign, output_benign, binary_search_steps=1)
            if image_adv_benign is None:
                image_adv_benign = attack(image_benign, output_benign, binary_search_steps=3)
            if image_adv_benign is None:
                os.remove(os.path.join(benign_dir, image_name))
                continue
        except:
            os.remove(os.path.join(benign_dir, image_name))
            continue
        
        image_adv_benign_pil = Image.fromarray((image_adv_benign * 255).astype(np.uint8))
        image_adv_benign_pil.save(os.path.join(adv_dir, image_name))

def calculate_scores(pervasive_dir, output_file_path, generage_false_iamge=False, wAP_threshold=0.09):
    dir_path_adv = os.path.join(pervasive_dir, "adv")
    image_name_list = os.listdir(dir_path_adv)

    if generage_false_iamge:
        false_sample_dir_path = './false_sample'
        if os.path.exists(false_sample_dir_path):
            shutil.rmtree(false_sample_dir_path)
        os.mkdir(false_sample_dir_path)
        os.mkdir(os.path.join(false_sample_dir_path, 'fp'))
        os.mkdir(os.path.join(false_sample_dir_path, 'fn'))

    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))
    '''
    kmodel = retina_resnet50()
    model = KerasResNet50RetinaNetModel(model=None, bounds=(0, 255))
    '''

    squz_fn = BitDepth(4)
    weighted_ap = WeightedAP(416, 416, wAP_threshold)

    scores_benign = []
    scores_adv = []
    with open(output_file_path, 'w') as out_file:
        out_file.write('{0},{1},{2},{3},{4}\n'.format('image_name', 'benign_wAP', 'adv_wAP', 'benign_mAP', 'adv_mAP'))

    for idx, image_name in enumerate(image_name_list):
        
        if image_name != 'fbbe1f4c-b1cf62c2_frame_00126.png':
            continue
        
        if idx % 100 == 0:
            print('idx : ', idx)
        image_name_noext = os.path.splitext(image_name)[0]

        temp_img_path_benign = os.path.join("../../../../../../../../datasets/bdd_parts/test_pervasive/benign", image_name)
        image_benign, shape = letterbox_image(
                data_format='channels_last', 
                fname=temp_img_path_benign
        ) 

        image_adv = np.array(Image.open(os.path.join(dir_path_adv, image_name))).astype(float) / 255.

        output_benign = model.predictions(image_benign)
        #print(output_benign)
        #draw = draw_letterbox(image, output_benign, shape[:2], model.class_names())
        #draw.save('benign.png')

        image_squz = squz_fn(image_benign)
        output_benign_squz = model.predictions(image_squz)

        #print(output_benign_squz)
        #draw = draw_letterbox(image_squz, output_benign_squz, shape[:2], model.class_names())
        #draw.save('benign_squz.png')

        wAP_score_benign = weighted_ap.distance_score(output_benign, output_benign_squz)
        mAP_score_benign = cal_mAP(output_benign, output_benign_squz, num_classes=80)
        #scores_benign_wAP.append(wAP_score_benign)
        #scores_benign_mAP.append(mAP_score_benign)

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
        mAP_score_adv = cal_mAP(output_adv, output_adv_squz, num_classes=80)
        #scores_adv.append(wAP_score_adv)
        #scores_adv_mAP.append(mAP_score_adv)
        '''
        print('wAP benign : ', wAP_score_benign)
        print('wAP adv : ', wAP_score_adv)
        print('mAP benign : ', mAP_score_benign)
        print('mAP adv : ', mAP_score_adv)
        '''
        with open(output_file_path, 'a') as out_file:
            out_file.write('{0},{1},{2},{3},{4}\n'.format(image_name_noext, str(wAP_score_benign), str(wAP_score_adv), str(mAP_score_benign), str(mAP_score_adv)))


        if wAP_score_benign > wAP_threshold:
            draw = draw_letterbox(image_benign, output_benign, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fp', image_name_noext + '_benign.png'))
            draw = draw_letterbox(image_squz, output_benign_squz, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fp', image_name_noext + '_benign_squz.png'))
            draw = draw_letterbox(image_adv, output_adv, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fp', image_name_noext + '_adv.png'))
            draw = draw_letterbox(image_adv_squz, output_adv_squz, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fp', image_name_noext + '_adv_squz.png'))
        if wAP_score_adv <= wAP_threshold:
            draw = draw_letterbox(image_benign, output_benign, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fn', image_name_noext + '_benign.png'))
            draw = draw_letterbox(image_squz, output_benign_squz, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fn', image_name_noext + '_benign_squz.png'))
            draw = draw_letterbox(image_adv, output_adv, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fn', image_name_noext + '_adv.png'))
            draw = draw_letterbox(image_adv_squz, output_adv_squz, shape[:2], model.class_names())
            draw.save(os.path.join(false_sample_dir_path, 'fn', image_name_noext + '_adv_squz.png'))
    

def show_letterbox_img(image_name):
    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))

    temp_img_path_benign = os.path.join("../../../../../../../../datasets/bdd_parts/test_pervasive/benign", image_name)
    image_benign, shape = letterbox_image(
            data_format='channels_last', 
            fname=temp_img_path_benign
    )
    output_benign = model.predictions(image_benign)
    draw = draw_letterbox(image_benign, output_benign, shape[:2], model.class_names())
    draw.save('letterbox_benign.png')

    temp_img_path_adv = os.path.join("../../../../../../../../datasets/bdd_parts/test_pervasive/adv", image_name)
    image_adv, shape = letterbox_image(
            data_format='channels_last', 
            fname=temp_img_path_adv
    )
    output_adv = model.predictions(image_adv)
    draw = draw_letterbox(image_adv, output_adv, shape[:2], model.class_names())
    draw.save('letterbox_adv.png')



if __name__ == "__main__":
    #generate_adv("/home/yantao/datasets/bdd_parts/test_pervasive/benign/", "/home/yantao/datasets/bdd_parts/test_pervasive/adv/")
    calculate_scores("/home/yantao/datasets/bdd_parts/test_pervasive", 'test_pervasive_result.csv', generage_false_iamge=True)
    #show_letterbox_img('de19448f-3e0c0324_frame_01323.png')