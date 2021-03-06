#!/bin/bash

IMGS_DIR='bdd10k_test' #'bdd10k_test', 'cityscapes_test_berlin'
ATTACK_MTD='cw_targetclsmiss' #'cw_targetclsmiss', 'noise_numberchange', 'bright_numberchange'
SQUZZEZ_MTD='bit_7'

CUDA_VISIBLE_DEVICES=3 python generate_adv.py --imgs-dir ${IMGS_DIR} --attack-mtd ${ATTACK_MTD}

CUDA_VISIBLE_DEVICES=3 python generate_squeeze.py ori --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
CUDA_VISIBLE_DEVICES=3 python generate_squeeze.py adv --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
CUDA_VISIBLE_DEVICES=3 python generate_detection_results.py --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
CUDA_VISIBLE_DEVICES=3 python calculate_wAP.py --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}
CUDA_VISIBLE_DEVICES=3 python evaluate.py --imgs-dir ${IMGS_DIR} --squeeze-type ${SQUZZEZ_MTD}