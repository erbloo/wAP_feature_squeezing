#!/bin/bash

IMGS_DIR='cityscapes_test_berlin'

CUDA_VISIBLE_DEVICES=3 python generate_adv.py --imgs-dir ${IMGS_DIR}