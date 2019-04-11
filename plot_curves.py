import glob
import pickle
import numpy as np
import csv
import cv2
import os
import shutil
import time
import matplotlib.pyplot as plt
from sklearn import metrics

import pdb

def arrange_data(ori_dic):
    line_list = []
    for key in ori_dic.keys():
        if not ori_dic[key]:
            continue
        value_list = ori_dic[key]
        class_name = key
        if len(class_name.split()) > 1:
            class_name = class_name.replace(' ', '_')
        for temp_value in value_list:
            temp_line = class_name + " " + str(temp_value['confidence']) + " " + str(int(temp_value['bbox'][0])) + " " + str(int(temp_value['bbox'][1])) + " " + str(int(temp_value['bbox'][2])) + " " + str(int(temp_value['bbox'][3]))
            line_list.append(temp_line)
    return line_list

def get_error(list1, list2, th):
    error_count = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    half_count = len(list1)
    for temp1 in list1:
        if temp1 > th:
            error_count += 1
            fp += 1
        else:
            tn += 1
    for temp2 in list2:
        if temp2 < th:
            error_count += 1
            fn += 1
        else:
            tp += 1

    er = float(error_count) / float(2 * half_count)
    re = float(tp) / float(tp + fn)
    if tp + fp == 0:
        pr = 0
    else:
        pr = float(tp) / float(tp + fp)

    return er, re, pr, tp, fp, tn, fn

def generate_false_samples(benign_tup_list, adv_tup_list, th, dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    os.mkdir(dir_path + '/fp')
    os.mkdir(dir_path + '/fn')

    for benign_tup in benign_tup_list:
        file_id = benign_tup[0]
        score = benign_tup[1]
        if score >= th:
            im1 = cv2.imread('/data/image/benign/boxed/' + file_id + '.png')
            im2 = cv2.imread('/data/image/squeezed/depth-5/benign_squeezed_boxed/benigh_' + file_id + '.png')
            im = np.concatenate((im1, im2), axis=1)
            cv2.imwrite(dir_path + '/fp/fp_' + file_id + '.png', im)
    for adv_tup in adv_tup_list:
        file_id = adv_tup[0]
        score = adv_tup[1]
        if score <= th:
            im1 = cv2.imread('/data/image/adv_car_only/boxed/adv_' + file_id + '.png')
            im2 = cv2.imread('/data/image/squeezed/depth-5/adv_squeezed_boxed/adv_adv_' + file_id + '.png')
            im = np.concatenate((im1, im2), axis=1)
            cv2.imwrite(dir_path + '/fn/fn_' + file_id + '.png', im)

def fp_fn_curve(gt_array, pd_array):
    gt_array = gt_array.astype('float')
    pd_array = pd_array.astype('float')

    ths = []
    fp = []
    fn = []
    n_total = gt_array.shape[0]

    th = 0.0
    while th <= 1.0:
        pd_b_array = np.zeros(pd_array.shape)
        for idx in range(n_total):
            if pd_array[idx] <= th:
                pd_b_array[idx] = 0
            else:
                pd_b_array[idx] = 1
        cm = metrics.confusion_matrix(gt_array, pd_b_array)
        
        ths.append(th)
        fp.append(cm[0][1])
        fn.append(cm[1][0])
        th += 0.01

    return np.asarray(fp).astype('float') / n_total, np.asarray(fn).astype('float') / n_total, np.asarray(ths)

def calculate_error_rate(score_benign_list, score_adv_list):
    th = 0.0001
    recall = 0
    precision = 0
    tpfptnfn = []
    min_er = [1, 0]
    while th <= 100:
        temp_er, temp_re, temp_pr, tp, fp, tn ,fn = get_error( score_benign_list, score_adv_list, th)
        if temp_er <= min_er[0]:
            min_er[0] = temp_er
            min_er[1] = th
            recall = temp_re
            precision = temp_pr
            tpfptnfn = np.asarray([tp, fp, tn, fn]) / (tp + fp + tn + fn)
        th += 0.01
    print('error rate: ', min_er)
    print('recall: ', recall)
    print('precision: ', precision)
    print('tpfptnfn: ', tpfptnfn)

def plot_curves(gt_array, pd_array, roc_file=None, fpfn_file=None):
    fpr, tpr, ths = metrics.roc_curve(gt_array, pd_array)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("RoC")
    plt.xlabel("fpr")
    plt.ylabel("tpr")   
    plt.show(block=False)
    if roc_file:
        plt.savefig(roc_file)

    fpr_1, fnr, ths = fp_fn_curve(gt_array, pd_array)
    plt.figure()
    plt.plot(fpr_1, fnr)
    plt.title("fp_fn curve")
    plt.xlabel("fpr")
    plt.ylabel("fnr")   
    plt.show()
    if fpfn_file:
        plt.savefig(fpfn_file)

def plot_curves_both(gt_array_wAP, pd_array_wAP, gt_array_mAP, pd_array_mAP, roc_file=None, fpfn_file=None):
    plt.figure()
    fpr, tpr, ths = metrics.roc_curve(gt_array_wAP, pd_array_wAP)
    plt.plot(fpr, tpr)
    fpr, tpr, ths = metrics.roc_curve(gt_array_mAP, pd_array_mAP)
    plt.plot(fpr, tpr)
    plt.title("RoC")
    plt.xlabel("fpr")
    plt.ylabel("tpr")   
    if roc_file:
        plt.savefig(roc_file)

    plt.figure()
    fpr_1, fnr, ths = fp_fn_curve(gt_array_wAP, pd_array_wAP)
    plt.plot(fpr_1, fnr)
    auc_wAP = metrics.auc(fpr_1, fnr)
    fpr_1, fnr, ths = fp_fn_curve(gt_array_mAP, pd_array_mAP)
    plt.plot(fpr_1, fnr)
    auc_mAP = metrics.auc(fpr_1, fnr)
    plt.title("fp_fn curve")
    plt.xlabel("fpr")
    plt.ylabel("fnr")   
    plt.show()
    if fpfn_file:
        plt.savefig(fpfn_file)

    print("wAP auc: ", auc_wAP)
    print("mAP auc: ", auc_mAP)

def main():
    result_path = "test_pervasive_result.csv"
    result_f = open(result_path, 'r')
    wAP_benign_list = []
    wAP_adv_list = []
    mAP_benign_list = []
    mAP_adv_list = []

    is_head = True
    for line in result_f:
        if is_head:
            is_head = False
            continue
        image_name, wAP_benign, wAP_adv, mAP_benign, mAP_adv = line.split(',')
        wAP_benign_list.append(float(wAP_benign))
        wAP_adv_list.append(float(wAP_adv))
        mAP_benign_list.append(1 - float(mAP_benign))
        mAP_adv_list.append(1 - float(mAP_adv))
    calculate_error_rate(wAP_benign_list, wAP_adv_list)
    calculate_error_rate(mAP_benign_list, mAP_adv_list)

    pd_wAP = np.asarray(wAP_benign_list + wAP_adv_list) / np.maximum(max(wAP_benign_list), max(wAP_adv_list))
    gt_wAP = np.asarray([0] * len(wAP_benign_list) + [1] * len(wAP_adv_list))

    pd_mAP = np.asarray(mAP_benign_list + mAP_adv_list) / np.maximum(max(mAP_benign_list), max(mAP_adv_list))
    gt_mAP = np.asarray([0] * len(mAP_benign_list) + [1] * len(mAP_adv_list))

    plot_curves_both(gt_wAP, pd_wAP, gt_mAP, pd_mAP, 'roc_bit5.png', 'fpfn_bit5.png')
    plot_curves(gt_wAP, pd_wAP, 'roc_wAP_bit5.png', 'fpfn_wAP_bit5.png')
    




if __name__ == "__main__":
    main()