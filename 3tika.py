import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os

def process_image(input_path, output_path):
    img_origin = tifffile.imread(input_path)
    
    # 画像をuint8に変換
    if img_origin.dtype != 'uint8':
        img_origin = img_origin.astype('uint8')
    #境界線検出
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(img_origin, kernel, iterations=1)
    new_border = cv2.subtract(dilated_mask, img_origin)
    
    #境界線を127で塗る
    img_filled = img_origin.copy()
    img_filled[new_border > 0] = 127
    tifffile.imwrite(output_path, img_filled)

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)
    print('finish')

    # 使用例
# input_folder = '/home/nakajima/work/Ecoli/code/lstmUnet/DataPrep/nd049_segmentation_64/Test/01_GT/SEG'  # 画像が保存されているフォルダのパス
input_folder = '/home/nakajima/work/Ecoli/code/Unet/removed_data/64_16/test/label'
output_folder = '/home/nakajima/work/Ecoli/code/lstmUnet/DataPrep/nd049_segmentation_64/Test2/01_GT/SEG'  # 処理後の画像を保存するフォルダのパス
# process_image(input_folder,output_folder)
process_images_in_folder(input_folder, output_folder)
