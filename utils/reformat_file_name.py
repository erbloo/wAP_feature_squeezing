import os
import pdb

dir_path = "/home/yantao/datasets/bdd_parts/cabc30fc-e7726578/benign/"

img_file_list = os.listdir(dir_path)
for img_file in img_file_list:
    img_file_noext = os.path.splitext(img_file)[0]
    file_idx = int((img_file_noext.split('_'))[-1])
    os.rename(os.path.join(dir_path, img_file), os.path.join(dir_path, 'frame_{0:05d}.png'.format(file_idx)))
    
