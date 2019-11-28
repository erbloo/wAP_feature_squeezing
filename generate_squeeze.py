from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/perceptron-benchmark')

from PIL import Image
import os
import shutil
import numpy as np 

from perceptron.defences.bit_depth import BitDepth


def main():
    video_name = 'bdd10k_test'
    input_dir = os.path.join("/home/yantao/workspace/datasets/wAP", video_name, "benign")
    output_dir = os.path.join("/home/yantao/workspace/datasets/wAP", video_name, "bit_5")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    image_name_list = os.listdir(input_dir)

    squz_fn = BitDepth(5)
    for _, image_name in enumerate(image_name_list):
        image_pil = Image.open(os.path.join(input_dir, image_name))
        image_np = np.array(image_pil) / 255.
        image_squz = squz_fn(image_np)
        image_squz_pil = Image.fromarray((image_squz * 255).astype(np.uint8))
        image_squz_pil.save(os.path.join(output_dir, image_name))

if __name__ == "__main__":
    main()
