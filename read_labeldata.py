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
from dist import Dist
from picture import Picture
from getshape import Getshape
import read_xml 
import re
import shutil
import random

pic = Picture()
gsh = Getshape()
dist1 = Dist()

import pickle
import pandas as pd
import json
import ast


def read_labeldata(input_path):
    with open(input_path, 'r') as file:
        data_str = file.read()
    data_dict = ast.literal_eval(data_str)

    img_size = data_dict['interval']['max']#画像サイズ
    # bg = data_dict['labels']['background']#背景
    foreground = data_dict['labels']['foreground']#seikai
    col = data_dict['labels']['col']
    
    img_size_np = np.array(img_size)
    # bg_np = np.array(bg)
    # label_np = np.array(foreground)
    label_np = np.array(col)
    print(data_dict['labels'].keys())

    img_size_np = img_size_np + np.ones(2,dtype=int)
    img_size_np[0], img_size_np[1]  = img_size_np[1],img_size_np[0]
    black_img = np.zeros(img_size_np)    
    for i in label_np:
        black_img[i[1],i[0]]=255

    return black_img

def main():
    path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label/'
    name = '8bit_nd049_S0_C1_T1'
    input_path = path + 'metadata/' + name +'.tiff.labeling'
    save_path = path + 'image/' + name +'_col.png'
    label_img = read_labeldata(input_path=input_path)
    cv2.imwrite(save_path, label_img)
    print('OK')

if __name__ == "__main__":
    main() 
   