from PIL import Image
import cv2
import numpy as np 
import os
import shutil
import imageio
import pdb

def dataset_img2video(dir_path, output_file):
    file_list = os.listdir(dir_path)
    file_list.sort()
    idx = 0
    while(idx < len(file_list)):
        file_path = file_list[idx]
        if (not file_path.endswith('.jpg')) or ("171206" not in file_path):
            del file_list[idx]
        else:
            idx += 1
    pdb.set_trace()
    imgs_path2video(dir_path, file_list, output_file)


def imgs_path2video(dir_path, path_list, output_file):
    video_writer = None
    for file_path in path_list:
        img = Image.open(os.path.join(dir_path, file_path))
        img_np = np.array(img)
        if video_writer == None:
            video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (img_np.shape[1], img_np.shape[0]))
        video_writer.write(img_np)

def video2imges(video_path, img_dir_folder, sub_folder=None):
    video_name = os.path.basename(video_path)
    video_name_noext = os.path.splitext(video_name)[0]
    img_dir = os.path.join(img_dir_folder, video_name_noext)
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if sub_folder is not None:
        img_child_dir = os.path.join(img_dir, sub_folder)
        if not os.path.isdir(img_child_dir):
            os.mkdir(img_child_dir)
        img_dir = img_child_dir

    vid = imageio.get_reader(video_path,  'ffmpeg')
    for idx, image in enumerate(vid.iter_data()):
        img_pil = Image.fromarray(image)
        img_pil.save(os.path.join(img_dir, "frame_%d.png" % idx))



if __name__ == "__main__":
    video2imges("/home/yantao/datasets/bdd_parts/cabc30fc-e7726578.mov", "/home/yantao/datasets/bdd_parts", sub_folder='benign')