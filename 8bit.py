import cv2
from PIL import Image
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from dist import Dist
from copy import copy

from picture import Picture

pic = Picture()
path = '/home/nakajima/work/Ecoli/data/nd049/nd049_S3' 
save_path = '/home/nakajima/work/Ecoli/data/nd049/8bit_nd049_S3_C1'
num, filelist = pic.get_picture(path)
h_list = []
hist_list = np.array(h_list)
for i in range(182,364):
    # print(f'file:{filelist[i]}')
    img = tifffile.imread(filelist[i])
    amin=np.amin(img)
    amax=np.amax(img)
    pixelsByte = ((img-amin)/(amax-amin))*255
    pixelsByte = np.clip(pixelsByte,0,255)
    pixelsByte = np.uint8(pixelsByte) # 符号なしバイトに変換
    tifffile.imsave(f'{save_path}/T{i-181:04d}.tiff',pixelsByte)


# #明るさの平均値と標準偏差を画面表示
# print("mean: " + str(np.mean(v)))
# print("std : " + str(np.std(v)))
# for i in range(int(num)):
#     plt.imsave(f'{save_path}{i}_new.png',tifffile.imread(filelist[i]))
# print(num)
# print(filelist)
print('OK')