import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import os
from PIL import Image
from scipy.stats import gaussian_kde
from getshape import Getshape
from search_image import Search
gsh = Getshape()
sarc = Search()



def make_overlap_image(truelabel, cover_image,combine_image_size,base_path):
    #overlapするときの関数
    #白色が正解ラベル
    #青色が予測結果
    true_label = np.where(truelabel == 0, 200, truelabel)
    background = np.zeros((combine_image_size[0], combine_image_size[1]), dtype=np.uint8)
    cover_image_color = np.array([background,background,cover_image]).transpose(1,2,0)
    true_label_color = np.array([true_label, true_label,true_label]).transpose(1,2,0)
    overlap_image = gsh.overlay_img(raw_img=true_label_color,proce_img=cover_image_color, num= [0.8,0.2,0] )
    overlap_image_rgb = overlap_image[:, :, [2, 1, 0]]
    overlap_image_rgb = np.where((overlap_image_rgb == [211,160,160]).all(axis=-1)[..., None],[160,255,255] ,overlap_image_rgb)#外れてる箇所
    overlap_image_rgb= np.where((overlap_image_rgb ==  [255,204,204]).all(axis=-1)[..., None], [0,0,255], overlap_image_rgb)#当たっている箇所
    overlap_image_rgb= np.where(overlap_image_rgb == 204, 255, overlap_image_rgb)#正解ラベルのみの箇所
    overlap_image_bgr = overlap_image_rgb[:, :, [2, 1, 0]]
    # cv2.imwrite(os.path.join(base_path,'overlap_image.png'),overlap_image_bgr)
    return overlap_image_bgr

def main():
    pred_file = '/home/nakajima/work/Ecoli/code/Unet/result/20250224_012822_256_16/image'
    label_file = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/scaledata/256_16/test_manual2/label'
    save_file = '/home/nakajima/work/Ecoli/code/Unet/result/20250224_012822_256_16/overlay'
    pred_image = sorted([f for f in os.listdir(pred_file) if f.endswith(('.png', '.tif', '.tiff'))])
    label_image = sorted([f for f in os.listdir(label_file) if f.endswith(('.png', '.tif', '.tiff'))])
    os.makedirs(save_file, exist_ok=True)
    num = 0
    for pred, label in zip(pred_image, label_image):
        cover_image = cv2.imread(os.path.join(pred_file,pred), cv2.IMREAD_GRAYSCALE)
        true_label = cv2.imread(os.path.join(label_file,label), cv2.IMREAD_GRAYSCALE)
        save_image = make_overlap_image(true_label, cover_image,[256,256], base_path=save_file)
        cv2.imwrite(os.path.join(save_file,f'overlap_image_{num:05d}.png'),save_image)
        num += 1
if __name__ == "__main__":
    main()