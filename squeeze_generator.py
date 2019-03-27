from __future__ import absolute_import
import sys
sys.path.append('/home/yantao/workspace/projects/baidu/aisec/perceptron')

from PIL import Image
import numpy as np 

from perceptron.defences.bit_depth import BitDepth

def _squeeze_image(image, squeeze_fn):
    image_squz = squeeze_fn(image)
    return image_squz

def generate_squeeze_image(img_path, output_path, mtd='bit_4'):
    if mtd == 'bit_4':
        squz_fn = BitDepth(4)
    img = Image.open(img_path)
    img_np = np.array(img).astype(np.float32) / 255.
    img_squz = _squeeze_image(img_np, squz_fn)
    img_squz_pil = Image.fromarray((img_squz * 255).astype(np.uint8))
    img_squz_pil.save(output_path)

def main():
    input_dir = "/home/yantao/datasets/bdd_parts/benign"
    output_dir = "/home/yantao/datasets/bdd_parts/benign"