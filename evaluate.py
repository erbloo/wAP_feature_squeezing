import glob
import argparse
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
    return

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
        temp_er, temp_re, temp_pr, tp, fp, tn ,fn = get_error(score_benign_list, score_adv_list, th)
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
    return min_er[1]

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
    return

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
    return

def temporal_simulate(th_wAP, wAP_benign_list, wAP_adv_list, th_mAP, mAP_benign_list, mAP_adv_list, r_age=3):
    np.random.seed(0)
    wAP_benign_single = np.where(np.array(wAP_benign_list) > th_wAP, 1, 0)
    wAP_adv_single = np.where(np.array(wAP_adv_list) > th_wAP, 1, 0)
    mAP_benign_single = np.where(np.array(mAP_benign_list) > th_mAP, 1, 0)
    mAP_adv_single = np.where(np.array(mAP_adv_list) > th_mAP, 1, 0)

    n_samples = len(wAP_benign_list)
    gt_single = np.random.rand(n_samples)
    gt_single = np.where(gt_single > 0.5, 1, 0)
    gt_temporal = _temporal_convert(gt_single, r_age)

    wAP_single = np.zeros(n_samples).astype(np.int)
    mAP_single = np.zeros(n_samples).astype(np.int)
    for idx in range(n_samples):
        if gt_single[idx] == 0:
            wAP_single[idx] = wAP_benign_single[idx]
            mAP_single[idx] = mAP_benign_single[idx]
        elif gt_single[idx] == 1:
            wAP_single[idx] = wAP_adv_single[idx]
            mAP_single[idx] = mAP_adv_single[idx]
        else:
            raise ValueError('Invalid element value, should be binary.')
    wAP_temporal = _temporal_convert(wAP_single, r_age)
    mAP_temporal = _temporal_convert(mAP_single, r_age)

    acc_wAP_single = _calculate_temporal_acc(wAP_single, gt_temporal)
    acc_mAP_single = _calculate_temporal_acc(mAP_single, gt_temporal)
    acc_wAP_temporal = _calculate_temporal_acc(wAP_temporal, gt_temporal)
    acc_mAP_temporal = _calculate_temporal_acc(mAP_temporal, gt_temporal)
    print('acc_wAP_single : {0:.4f}'.format(acc_wAP_single))
    print('acc_mAP_single : {0:.4f}'.format(acc_mAP_single))
    print('acc_wAP_temporal : {0:.4f}'.format(acc_wAP_temporal))
    print('acc_mAP_temporal : {0:.4f}'.format(acc_mAP_temporal))
    return
    
def _calculate_temporal_acc(pd, gt):
    assert len(pd.shape) <= 1 and len(pd.shape) == len(gt.shape)
    num = pd.shape[0]
    corrects = 0
    for temp_pd, temp_gt in zip(pd, gt):
        if temp_pd == temp_gt:
            corrects += 1
    return float(corrects) / float(num)

def _temporal_convert(gt_single, r_age):
    assert len(gt_single.shape) <= 1
    n_samples = gt_single.shape[0]
    gt_temporal = np.zeros(n_samples).astype(np.int)
    for idx in range(n_samples - 2):
        temp_flag = False
        for sub_idx in range(r_age):
            if gt_single[idx + sub_idx] != 1:
                break
            if sub_idx == r_age - 1:
                temp_flag = True
        if temp_flag:
            for sub_idx in range(r_age):
                gt_temporal[idx + sub_idx] = 1
    return gt_temporal

def main(args):
    result_path = "{1}_{0}.csv".format(args.squeeze_type, args.imgs_dir)
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
    th_wAP = calculate_error_rate(wAP_benign_list, wAP_adv_list)
    th_mAP = calculate_error_rate(mAP_benign_list, mAP_adv_list)

    pd_wAP = np.asarray(wAP_benign_list + wAP_adv_list) / np.maximum(max(wAP_benign_list), max(wAP_adv_list))
    gt_wAP = np.asarray([0] * len(wAP_benign_list) + [1] * len(wAP_adv_list))

    pd_mAP = np.asarray(mAP_benign_list + mAP_adv_list) / np.maximum(max(mAP_benign_list), max(mAP_adv_list))
    gt_mAP = np.asarray([0] * len(mAP_benign_list) + [1] * len(mAP_adv_list))

    plot_curves_both(gt_wAP, pd_wAP, gt_mAP, pd_mAP, 'roc_{0}.png'.format(args.squeeze_type), 'fpfn_{0}.png'.format(args.squeeze_type))
    plot_curves(gt_wAP, pd_wAP, 'roc_wAP_{0}.png'.format(args.squeeze_type), 'fpfn_wAP_{0}.png'.format(args.squeeze_type))

    temporal_simulate(th_wAP, wAP_benign_list, wAP_adv_list, th_mAP, mAP_benign_list, mAP_adv_list)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot.")
    parser.add_argument('--imgs-dir', type=str, default='bdd10k_test')
    parser.add_argument('--squeeze-type', type=str, default='bit_7')
    args = parser.parse_args()
    main(args)