# from stardist.models import StarDist2D
# from csbdeep.utils import normalize
# from stardist import random_label_cmap

import numpy as np
import os
import PIL
import PIL.Image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow_datasets as tfds
import shutil 
import stackview
import matplotlib.pyplot as plt
from skimage.data import human_mitosis
# from keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import statistics
import warnings

import tifffile as tiff
import cv2
import inspect
import pandas as pd
from stardist.matching import matching
import csv
import re
import glob
import math


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # 親ディレクトリをパスに追加
# sys.path.append("..")
from picture import Picture


def get_sorted_image(folder_path):
    pic = Picture()
    _, pic_list = pic.get_picture(folder_path)
    # image_paths = glob.glob(os.path.join(folder_path, '*.*'))
    # # ファイル名で昇順に並べ替える（降順にしたい場合は reverse=True を追加）
    # sorted_image_paths = sorted(image_paths, key=lambda x: os.path.basename(x))
    return pic_list

#画像読み込み
def load_image(path):
    try:
        input_image = tiff.imread(path)
    except:
        input_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return input_image


def parse_and_write_metrics(file_path, metrics_str):
    # メトリクス文字列からキーと値を抽出
    pattern = r'(\w+)=([-+]?\d*\.?\d+)'  # キー=値 の形式を正規表現で抽出
    parsed_metrics = re.findall(pattern, metrics_str)

    # キーと数値を分離
    keys = [key for key, _ in parsed_metrics]
    values = [float(value) if '.' in value else int(value) for _, value in parsed_metrics]

    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(keys)
        writer.writerow(values)

def cal(label_list, predict_list, save_path):
    # 予測
    for i in range(len(label_list)):
        label = tiff.imread(label_list[i])
        if label.ndim != 2:
            label = cv2.imread(label_list[i], cv2.IMREAD_GRAYSCALE)
        predict = tiff.imread(predict_list[i])
        predict = predict.astype(np.uint8)
        metrics =  matching(label, predict)
        
        parse_and_write_metrics(save_path+'/metrics2.csv', str(metrics))
        # cv2.imwrite(os.path.join(save_path, 'image',f'{i + 1:05d}.tif'), predict[0])
        # cv2.imwrite(os.path.join(save_path, 'image_normalize',f'{i + 1:05d}.png'),  (predict[0] * (255/(np.max(predict[0])+1))).astype(np.uint8))

def acc_csv(path):
    input_file =  path + "/metrics2.csv" 
    output_file = path + "/non_0_metrics2.csv" 
    df = pd.read_csv(input_file)

    # n_pred が非0の行をフィルタリング
    filtered_df = df[df['n_pred'] != 0].copy()
    row_indices = filtered_df.index.tolist()
    filtered_df.insert(0, "extracted_rows", row_indices)

    mean_values = filtered_df.mean(axis=0)
    empty_row = {col: "" for col in filtered_df.columns}
    filtered_df = pd.concat([filtered_df, pd.DataFrame([empty_row]), pd.DataFrame([mean_values])], ignore_index=True)

    # 新しいCSVファイルに保存
    filtered_df.to_csv(output_file, index=False)


def f1_iou(pred,target, num_classes):
        warnings.simplefilter('ignore')
        # print(num_classes)
        if num_classes < 2:
            num_classes = 2
        ious = [0]*(num_classes+2)
        no_nan_ious = [0]*(num_classes+2)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        array_ious = []
        anyious =0
        # print(num_classes, len(ious))

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = np.sum(pred_inds & target_inds)
            union = np.sum(pred_inds | target_inds)
            
            if union == 0:
                # array_ious.append(float('nan'))  # If there is no ground truth, IoU is undefined
                array_ious.append(np.nan)  # If there is no ground truth, IoU is undefined
                ious[cls] = array_ious[cls]*100#クラスclsのIoU
            else:
                array_ious.append(intersection / union)
                ious[cls] = array_ious[cls]*100#クラスclsのIoU
        for i in range(num_classes-1):
                anyious += array_ious[i+1]
        anyious = anyious/(num_classes-1)
        ious[num_classes] = anyious
        ious[num_classes+1] = statistics.mean(array_ious)*100#全クラスのmIoU
        valid_ious = [iou for iou in array_ious if not np.isnan(iou)]
        no_nan_ious[num_classes] = statistics.mean(valid_ious) * 100 if valid_ious else float('nan') 
        
        # f1スコアの計算
        # if num_classes == 2:
        #     f1score = [recall_score(target,pred)*100,
        #             precision_score(target,pred)*100,
        #             f1_score(target,pred)*100]
        # else:
        f1score = [np.nan,np.nan,np.nan ]
        return ious, f1score

def cal_main(pred_path, label_path,save_path):
    ins_save_path = os.path.join(save_path,'ins_score.csv')
    sem_save_path = os.path.join(save_path,'sem_score.csv')
    total_ins_iou = []
    total_sem_iou = []
    for pred, label in zip(pred_path, label_path):
        pred_img = load_image(pred)
        label_img = load_image(label)
        iou, non_nan_iou =  f1_iou(pred_img,label_img,len(np.unique(label_img)))
        semantic = np.where(pred_img > 0, 1, 0)
        sem_label = np.where(label_img > 0, 1, 0)
        # _, semantic = cv2.threshold(pred_img.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
        # _, sem_label = cv2.threshold(label_img.astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
        sem_iou,  sem_non_nan_iou =  f1_iou(semantic,sem_label,2)
        save_score(iou,ins_save_path)
        save_score(sem_iou, sem_save_path)
        total_ins_iou.append(iou)
        total_sem_iou.append(sem_iou)
    analyze_matrices(np.array(total_ins_iou),np.array(total_sem_iou),save_path)

def save_score(score,save_path):
    with open(save_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(score)

def analyze_matrices(ins_score, sem_score,save_path):
    save_path_ins = os.path.join(save_path,'no_nan_ins_score.csv')
    save_path_sem = os.path.join(save_path,'no_nan_sem_score.csv')
    np.set_printoptions(suppress=True, precision=8)
    no_nan_sem_score = sem_score[~np.isnan(sem_score).any(axis=1), :]
    no_nan_ins_score = [row for row in ins_score if not any(math.isnan(val) for val in row)]
    
    save_score(['back_ground', 'class1', 'nan', 'mean'],save_path_sem)
    for score in no_nan_sem_score:
        save_score(score,save_path_sem)
    for score2 in no_nan_ins_score:
        save_score(score2,save_path_ins)

    if no_nan_sem_score.shape[0] > 0:  # 行が存在する場合のみ計算
        save_score(['mean_iou'],save_path_sem)
        save_score(np.mean(no_nan_sem_score, axis=0),save_path_sem)
        save_score(['var'],save_path_sem)
        save_score(np.var(no_nan_sem_score, axis=0),save_path_sem)
        save_score(['std'],save_path_sem)
        save_score(np.std(no_nan_sem_score, axis=0),save_path_sem)
    

    if no_nan_ins_score:
        last_elements = []         # 各行の後ろから1番目の要素を収集
        second_last_elements = []  # 各行の後ろから2番目の要素を収集
        
        for row in no_nan_ins_score:
            if len(row) >= 1:
                last_elements.append(row[-1])        # 後ろから1番目の要素を取得
            if len(row) >= 2:
                second_last_elements.append(row[-2]) # 後ろから2番目の要素を取得

    save_score(['class_mean_iou','class_var','class_std'],save_path_ins)
    save_score([np.mean(second_last_elements),np.var(second_last_elements),np.std(second_last_elements)],save_path_ins)
    save_score(['mean_iou','var','std'],save_path_ins)
    save_score([np.mean(last_elements),np.var(last_elements),np.std(last_elements)],save_path_ins)
    


def main():
    base_path = '/home/nakajima/work/Ecoli/code/stardist_local/mymodel/2D_versatile_fluo_256_trainglay_no_pro_img'
    predict_name = 'pred_image'
    predict_path = os.path.join(base_path, predict_name)
    label_path = '/home/nakajima/work/Ecoli/code/stardist_local/scaledata/256_16/test_manual2_class/label'
    pic = Picture()
    _, predict_list = pic.get_picture(predict_path)
    _, label_list = pic.get_picture(label_path) 
    cal_main(predict_list,label_list,base_path)
if __name__ == '__main__':
    main()