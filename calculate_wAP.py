from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')

from PIL import Image
import numpy as np 
import os
import pdb
from collections import defaultdict
import pickle

from perceptron.utils.image import letterbox_image, draw_letterbox
from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
from perceptron.zoo.yolov3.model import YOLOv3
from perceptron.attacks.carlini_wagner import CarliniWagnerLinfAttack
from perceptron.utils.criteria.detection import TargetClassMiss, WeightedAP
from perceptron.defences.bit_depth import BitDepth

def cal_mAP(gt_dict, pred_dict, num_classes, th_conf=0.5):
        class_pred_tmp = {}
        for class_name in range(num_classes):
            class_pred_tmp[class_name] = []

        out_boxes = pred_dict['boxes']
        out_scores = pred_dict['scores']
        out_classes = pred_dict['classes']

        for idx in range(len(out_classes)):
            predicted_class = out_classes[idx]
            box = out_boxes[idx]
            score = out_scores[idx] 
            class_pred_tmp[predicted_class].append({"confidence": str(score), "bbox": box})

        sum_AP = 0.0
        gt_counter_per_class, gt_objects = preprocess_ground_truth(gt_dict)
        for class_name in sorted(gt_counter_per_class):
            prediction_data = class_pred_tmp[class_name]
            nd = len(prediction_data)
            tp = [0] * nd
            fp = [0] * nd
            for idx, prediction in enumerate(prediction_data):
                ovmax = -1
                match_idx = -1
                bb = prediction["bbox"]
                for i, obj in enumerate(gt_objects):
                    if(obj["class_name"]) == class_name:
                        bbgt = obj["bbox"]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                match_idx = i
                if ovmax >= th_conf:
                    if not gt_objects[match_idx]['used']:
                        tp[idx] = 1
                        gt_objects[match_idx]['used'] = True
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val           
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            
            ap, _, _ = voc_ap(rec, prec)
            sum_AP += ap

        mAP = sum_AP / len(gt_counter_per_class)
        return mAP

def preprocess_ground_truth(ground_truth):
    gt_counter_per_class = defaultdict(int)
    objects = []
    for idx in range(len(ground_truth['classes'])):
        class_name = ground_truth['classes'][idx]
        bbox = ground_truth['boxes'][idx]
        obj = {}
        obj["bbox"] = bbox
        obj["used"] = False
        obj["class_name"] = class_name
        objects.append(obj)
        gt_counter_per_class[class_name] += 1

    return gt_counter_per_class, objects 
        

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
        This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
        This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
        The Average Precision (AP) is the area under the curve
        (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def main_from_pickle():
    video_name = 'cabc30fc-e7726578'
    pickle_benign_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "benign_det_results")
    pickle_adv_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_det_results")
    pickle_benign_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "bit_5_det_results")
    pickle_adv_squz_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "adv_bit_5_det_results")
    image_name_list = os.listdir(pickle_benign_dir)

    weighted_ap = WeightedAP(416, 416, 0.0001)

    scores_benign = []
    scores_adv = []
    csv_file_name = video_name + '_bit5.csv'
    with open(csv_file_name, 'w') as out_file:
        out_file.write('{0},{1},{2},{3},{4}\n'.format('image_name', 'benign_wAP', 'adv_wAP', 'benign_mAP', 'adv_mAP'))

    for idx, image_name in enumerate(image_name_list):
        '''
        if idx % 10 == 0:
            print('idx : ', idx)
        '''
        image_name_noext = os.path.splitext(image_name)[0]

        with open(os.path.join(pickle_benign_dir, image_name_noext + '.pkl'), 'rb') as handle:
            output_benign = pickle.load(handle)
        #print(output_benign)
        #draw = draw_letterbox(image, output_benign, shape[:2], model.class_names())
        #draw.save('benign.png')

        with open(os.path.join(pickle_benign_squz_dir, image_name_noext + '.pkl'), 'rb') as handle:
            output_benign_squz = pickle.load(handle)
        #print(output_benign_squz)
        #draw = draw_letterbox(image_squz, output_benign_squz, shape[:2], model.class_names())
        #draw.save('benign_squz.png')

        wAP_score_benign = weighted_ap.distance_score(output_benign, output_benign_squz)
        mAP_score_benign = cal_mAP(output_benign, output_benign_squz, num_classes=80)
        #scores_benign_wAP.append(wAP_score_benign)
        #scores_benign_mAP.append(mAP_score_benign)

        with open(os.path.join(pickle_adv_dir, image_name_noext + '.pkl'), 'rb') as handle:
            output_adv = pickle.load(handle)
        #print(output_adv)
        #draw = draw_letterbox(image_adv, output_adv, shape[:2], model.class_names())
        #draw.save('adv.png')

        with open(os.path.join(pickle_adv_squz_dir, image_name_noext + '.pkl'), 'rb') as handle:
            output_adv_squz = pickle.load(handle)
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
        with open(csv_file_name, 'a') as out_file:
            out_file.write('{0},{1},{2},{3},{4}\n'.format(image_name_noext, str(wAP_score_benign), str(wAP_score_adv), str(mAP_score_benign), str(mAP_score_adv)))

def main():
    video_name = 'cabc30fc-e7726578'
    dir_path = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "benign")
    image_name_list = os.listdir(dir_path)

    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))
    
    attack = CarliniWagnerLinfAttack(model, criterion=TargetClassMiss(2))

    squz_fn = BitDepth(4)
    weighted_ap = WeightedAP(416, 416, 0.0001)

    scores_benign = []
    scores_adv = []
    csv_file_name = video_name + '.csv'
    with open(csv_file_name, 'w') as out_file:
        out_file.write('{0},{1},{2},{3},{4}\n'.format('image_name', 'benign_wAP', 'adv_wAP', 'benign_mAP', 'adv_mAP'))

    for idx, image_name in enumerate(image_name_list):
        if idx % 10 == 0:
            print('idx : ', idx)
        image_name_noext = os.path.splitext(image_name)[0]

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
        mAP_score_benign = cal_mAP(output_benign, output_benign_squz, num_classes=80)
        #scores_benign_wAP.append(wAP_score_benign)
        #scores_benign_mAP.append(mAP_score_benign)

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
        mAP_score_adv = cal_mAP(output_adv, output_adv_squz, num_classes=80)
        #scores_adv.append(wAP_score_adv)
        #scores_adv_mAP.append(mAP_score_adv)

        print('wAP benign : ', wAP_score_benign)
        print('wAP adv : ', wAP_score_adv)
        print('mAP benign : ', mAP_score_benign)
        print('mAP adv : ', mAP_score_adv)

        with open(csv_file_name, 'a') as out_file:
            out_file.write('{0},{1},{2},{3},{4}\n'.format(image_name_noext, str(wAP_score_benign), str(wAP_score_adv), str(mAP_score_benign), str(mAP_score_adv)))


    





if __name__ == "__main__":
    main_from_pickle()