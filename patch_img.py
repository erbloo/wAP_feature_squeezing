from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from perceptron.utils.image import letterbox_image, draw_letterbox
import pickle
import pdb

def get_patch_img():
    video_name = 'cabc30fc-e7726578'
    img_path = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "benign", 'frame_01173.png')
    image, shape = letterbox_image(
            data_format='channels_last', 
            fname=img_path
    ) 

    plt.figure()
    plt.imshow(np.array(image * 255).astype(np.uint8))
    plt.show()
    pdb.set_trace()

    # cabc30fc-e7726578 : 1173~1192
    patch_dic = {
        'frame_01173' : [227, 249, 237, 283],
        'frame_01174' : [227, 249, 237, 283],
        'frame_01175' : [227, 249, 237, 283],
        'frame_01176' : [227, 249, 237, 283],
        'frame_01177' : [227, 249, 237, 283],
        'frame_01178' : [227, 249, 237, 283],
        'frame_01179' : [227, 249, 237, 283],
        'frame_01180' : [227, 249, 237, 283],
        'frame_01181' : [227, 249, 237, 283],
        'frame_01182' : [227, 249, 237, 283],
        'frame_01183' : [227, 249, 237, 283],
        'frame_01184' : [227, 249, 237, 283],
        'frame_01185' : [227, 249, 237, 283],
        'frame_01186' : [227, 249, 237, 283],
        'frame_01187' : [227, 249, 237, 283],
        'frame_01188' : [227, 249, 237, 283],
        'frame_01189' : [227, 249, 237, 283],
        'frame_01190' : [227, 249, 237, 283],
        'frame_01191' : [227, 249, 237, 283],
        'frame_01192' : [227, 249, 237, 283],
    }


    for idx in range(209, 229):
        img_name_noext = 'frame_{0:05d}'.format(idx)
        patch_bbox = patch_dic[img_name_noext]
        img_path = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "benign", img_name_noext + '.png')
        image, shape = letterbox_image(
                data_format='channels_last', 
                fname=img_path
        ) 
        for idx_h in range(shape[0]):
            for idx_w in range(shape[1]):
                if idx_h >= patch_bbox[0] and idx_h <= patch_bbox[2] and idx_w >= patch_bbox[1] and idx_w <= patch_bbox[3]:
                    image[idx_h, idx_w, 0] = 0.5
                    image[idx_h, idx_w, 1] = 0.5
                    image[idx_h, idx_w, 2] = 0.5
        img_out_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_out_path = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "patch", img_name_noext + '.png')
        img_out_pil.save(img_out_path)

def get_patch_adv():
    from perceptron.utils.image import letterbox_image, draw_letterbox
    from perceptron.models.detection.keras_yolov3 import KerasYOLOv3Model
    from perceptron.zoo.yolov3.model import YOLOv3
    from perceptron.attacks.vanish_patch import PatchVanishAttack
    from perceptron.utils.criteria.detection import TargetClassMiss, RegionalTargetClassMiss
    from perceptron.defences.bit_depth import BitDepth
    video_name = 'cabc30fc-e7726578'

    # cabc30fc-e7726578 : 1173~1192
    patch_dic = {
        'frame_01173' : [227, 249, 237, 283],
        'frame_01174' : [227, 249, 237, 283],
        'frame_01175' : [227, 249, 237, 283],
        'frame_01176' : [227, 249, 237, 283],
        'frame_01177' : [227, 249, 237, 283],
        'frame_01178' : [227, 249, 237, 283],
        'frame_01179' : [227, 249, 237, 283],
        'frame_01180' : [227, 249, 237, 283],
        'frame_01181' : [227, 249, 237, 283],
        'frame_01182' : [227, 249, 237, 283],
        'frame_01183' : [227, 249, 237, 283],
        'frame_01184' : [227, 249, 237, 283],
        'frame_01185' : [227, 249, 237, 283],
        'frame_01186' : [227, 249, 237, 283],
        'frame_01187' : [227, 249, 237, 283],
        'frame_01188' : [227, 249, 237, 283],
        'frame_01189' : [227, 249, 237, 283],
        'frame_01190' : [227, 249, 237, 283],
        'frame_01191' : [227, 249, 237, 283],
        'frame_01192' : [227, 249, 237, 283],
    }

    benign_output_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "patch_det")
    adv_output_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "patch_adv_det")
    benign_squz_output_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "patch_bit4_det")
    adv_squz_output_dir = os.path.join("/home/yantao/datasets/bdd_parts", video_name, "patch_adv_bit4_det")

    kmodel = YOLOv3()
    kmodel.load_weights('/home/yantao/workspace/projects/baidu/aisec/perceptron/perceptron/zoo/yolov3/model_data/yolov3.h5')
    model = KerasYOLOv3Model(kmodel, bounds=(0, 1))

    squz_fn = BitDepth(4)

    for idx in range(209, 229):
        img_name_noext = 'frame_{0:05d}'.format(idx)
        patch_bbox = patch_dic[img_name_noext]
        img_path = os.path.join("../../../../../../../../datasets/bdd_parts", video_name, "patch", img_name_noext + '.png')
        image, shape = letterbox_image(
                data_format='channels_last', 
                fname=img_path
        ) 

        output_benign = model.predictions(image)
        with open(os.path.join(benign_output_dir, img_name_noext + '.pkl'), 'wb') as f:
            pickle.dump(output_benign, f, protocol=pickle.HIGHEST_PROTOCOL)

        image_squz = squz_fn(image)
        output_benign_squz = model.predictions(image_squz)
        with open(os.path.join(benign_squz_output_dir, img_name_noext + '.pkl'), 'wb') as f:
            pickle.dump(output_benign_squz, f, protocol=pickle.HIGHEST_PROTOCOL)

        attack = PatchVanishAttack(model, criterion=RegionalTargetClassMiss(2, patch_bbox))
        
        draw = draw_letterbox(image, output_benign, shape[:2], model.class_names())
        draw.save('letterbox_{0}.png'.format(idx))

        image_adv = attack(image, mask=patch_bbox, binary_search_steps=1)
        '''
        mask_img = np.zeros((image.shape))
        mask_img[patch_bbox[0] : patch_bbox[2] + 1, patch_bbox[1] : patch_bbox[3] + 1, :] = 1.0
        for idx_0 in range(mask_img.shape[0]):
            for idx_1 in range(mask_img.shape[1]):
                for idx_2 in range(mask_img.shape[2]):
                    if mask_img[idx_0, idx_1, idx_2] == 0:
                        image_adv[idx_0, idx_1, idx_2] = image[idx_0, idx_1, idx_2]
        '''
        if image_adv is None:
            image_adv = attack(image, mask=patch_bbox, binary_search_steps=5)
        if image_adv is None:
            continue

        output_adv = model.predictions(image_adv)
        with open(os.path.join(adv_output_dir, img_name_noext + '.pkl'), 'wb') as f:
            pickle.dump(output_adv, f, protocol=pickle.HIGHEST_PROTOCOL)

        draw = draw_letterbox(image_adv, output_adv, shape[:2], model.class_names())
        draw.save('letterbox_adv_{0}.png'.format(idx))

        image_adv_squz = squz_fn(image_adv)
        output_adv_squz = model.predictions(image_adv_squz)
        with open(os.path.join(adv_squz_output_dir, img_name_noext + '.pkl'), 'wb') as f:
            pickle.dump(output_adv_squz, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    get_patch_img()
    #get_patch_adv()

