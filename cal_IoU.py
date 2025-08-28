import cv2
from PIL import Image
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from copy import copy
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
from skimage import measure
import csv
import statistics
from search_image import Search
from picture import Picture
from getshape import Getshape
import read_xml 
import re
import shutil
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from scipy import stats
import warnings

pic = Picture()
gsh = Getshape()
dist1 = Search()
random.seed(0)

def read_image(img_path):
    try:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    except:
        image = tifffile.imread(img_path)  # グレー画像として読み込む
    return image

def f1_IoU(pred, true):
    num_classes = 2
    ious = [0]*(num_classes+1)
    array_ious = list()
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        true_inds = (true == cls)
        intersection = (pred_inds & true_inds).sum().item()
        union = (pred_inds | true_inds).sum().item()
        
        if union == 0:
            array_ious.append(float('nan'))  # If there is no ground truth, IoU is undefined
            ious[cls] = array_ious[cls]*100#クラスclsのIoU
        else:
            array_ious.append(intersection / union)
            ious[cls] = array_ious[cls]*100#クラスclsのIoU

    ious[num_classes] = (array_ious[1])
    # ious[num_classes] = (array_ious[1]+array_ious[2])/2
    # ious[num_classes+1] = statistics.mean(array_ious)#全クラスのmIoU
    ious[num_classes] = statistics.mean(array_ious)*100#全クラスのmIoU
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    f1score = [recall_score(true_flat,pred_flat,average='binary',pos_label=1)*100,
            precision_score(true_flat,pred_flat,average='binary',pos_label=1)*100,
            f1_score(true_flat,pred_flat,average='binary',pos_label=1)*100]
    return ious, f1score



def cal_iou(img_path, lab_path, save_path):
    num1, img_filename =pic.get_picture(img_path)
    num2, lab_filename = pic.get_picture(lab_path)
    total_iou = []
    total_f1score = []
    for i in range(len(lab_filename)):
        img= read_image(img_filename[i])
        lab= read_image(lab_filename[i])
        print(img)
        print(lab)
        iou,f1score = f1_IoU(lab,img)
        total_iou.append(iou)
        total_f1score.append(f1score)
    class1 =  statistics.mean(total_iou[:][0])
    class2 = statistics.mean(total_iou[:][1])
    mean = statistics.mean(total_iou[:][2])
    sigma1 =statistics.pvariance(total_iou[:][0])
    sigma2 =statistics.pvariance(total_iou[:][1])
    sigma_mean =statistics.pvariance(total_iou[:][2])
    
    with open(save_path + '/' + 'score.txt', 'w') as f:
        for i in range(len(lab_filename)):
            f.write(f'class1: {total_iou[i][0]},'
                    f'class2: {total_iou[i][1]},'
                    f'mean_score: {total_iou[i][2]},'
                    f'recall: {total_f1score[i][0]},'
                    f'precision: {total_f1score[i][1]},'
                    f'f1-measure: {total_f1score[i][2]},'
                    '\n')
        f.write(f'mean class1 {class1},'
                f'mean class2 {class2},'
                f'mean  {mean},'
                f'sig1 {sigma1},'
                f'sig2 {sigma2},'
                f'sig {sigma_mean},'             
                )
        
def main(pred_path):
    """
    20241203_191357_64_16_con
    20241204_025735_64_16
    20241204_033400_64_16
    20241204_140515_64_16
    """

    label_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_label.tif'
    common_path = '/home/nakajima/work/Ecoli/code/Unet/result/'
    path =  common_path + pred_path
    cal_iou(img_path= path + '/combined_image_or.png', lab_path= label_path, save_path= path)

if __name__ == "__main__":
    path = '20241204_224230_64_16'
    main(pred_path=path)
