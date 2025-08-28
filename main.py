import cv2
from PIL import Image
import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import matplotlib.pyplot as plt
from copy import copy
# from skimage.filters.rank import entropy
from skimage.morphology import disk, ball

from dist import Dist
from picture import Picture
from getshape import Getshape

pic = Picture()
gsh = Getshape()
# raw_path = '/home/nakajima/work/Ecoli/data/nd049_S0_C1/nd049_S0_C1_T20.tiff' 
raw_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S0_C1_T0.tiff'
# raw_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/label/S1_C1_T0_elip_label.tiff'#raberu 
path = '/home/nakajima/work/Ecoli/data/new_nd049_s0_shape/lap_5_/temp_match/temp_circle_128.png' 
# raw_path = '/home/nakajima/work/Ecoli/data/new_nd049_s0_shape/lap_5_/processing_/bil3010x4_lap_otsu.png' 
raw_path2 = '/home/nakajima/work/Ecoli/data/nd049_S0_C1/nd049_S0_C1_T21' 
Save_path = '/home/nakajima/work/Ecoli/code/lstmUnet/DataPrep/nd049_segmentation_small/Train/01'
# Num, Filelist = pic.get_picture(path, Save_path)
# Num, Rawfilelist = pic.get_picture(raw_path, Save_path)
# Num, Rawfilelist2 = pic.get_picture(raw_path2, Save_path)


import math
def main(num,filelist,rawfilelist,save_path):
    thres = 0.3
    num = 40
    p_path, o_path = gsh.make_dir(save_path)
    # img = cv2.imread(rawfilelist)
    img = tifffile.imread(rawfilelist)
    imgs = gsh.split(img)
    print(len(imgs))
    for i in range(len(imgs)):
        save_path = save_path  + str(i).zfill(3)
        # print(save_path)
        cv2.imwrite(f'{save_path}.tiff', imgs[i])
    # for i  in range(4):
    #     img = gsh.imgfilter(img, 'bil', 0)

    # img = gsh.edge(img, 'lap',[0,0,5])
    # img = gsh.threshold(img, 'otsu')

    # img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength,maxLineGap)
    # print(lines)
    # print(len(lines))
    # for i in range(len(lines)):
    #     x1 = lines[i][0][0]
    #     y1 = lines[i][0][1]
    #     x2 = lines[i][0][2]
    #     y2 = lines[i][0][3]
    #     cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),3)
    
    # cv2.imwrite(f'{save_path}P_min100_max100_the100.tiff',img2)
    # # img1 = img[0:1024,110:7022]
    # # ハフ変換
    # lines = cv2.HoughLines(img, 3, np.pi / 180, 100)
    # print(lines)
    # print(len(lines))
    # # 結果表示用の画像を作成
    # img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # line_length = 10
    # for line in lines:
    #     rho = line[0][0]
    #     theta = line[0][1]
    #     a = math.cos(theta)
    #     b = math.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     cv2.line(
    #         img2, 
    #         (int(x0 - line_length * b), int(y0 + line_length * a)), 
    #         (int(x0 + line_length * b), int(y0 - line_length * a)), 
    #         (0, 0, 255), thickness=2, lineType=cv2.LINE_4 )

    # cv2.imshow("result", img2)

    # cv2.waitKey(5000)

    # for rho,theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)
    # cv2.imwrite('houghlines1_100.png',img2)
    # for i in range(4):
    #     img = gsh.imgfilter(img,'bil',0)
    # img = gsh.edge(img,'lap',[0,0,5])    
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV,7,8)
    # print(img)
    # print(img.shape)
    # img = np.array(img).flatten()
    
    # mean = img.mean()             # 平均値
    # std = np.std(img)             # 標準偏差
    # median = np.median(img)       # 中央値

    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))

    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    # res = gsh.template_matching(temp, img ,thres)

    # gsh.hist(img,histname)
    kernel1 = np.ones((2,2),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1, iterations = 1)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1, iterations = 2)
    # col_img = color_img(bin_img)
    name='/home/nakajima/work/Ecoli/data/new_nd049_s0_shape/lap_5_/processing_/' 
    name = p_path +'bil3_30_10_x4_lap_otsu'
    # cv2.imwrite(f'{name}.tiff', img)
    # print("OK")

    

if __name__ == "__main__":
    main(0, path, raw_path, Save_path) 
    # file1 = '/home/nakajima/work/Ecoli/data/new_nd049_s0_shape/lap_5_/processing_/bil3010x4_lap_otsu.png'
    # file2 = '/home/nakajima/work/Ecoli/data/new_nd049_s0_shape/lap_5_/processing_/bil3010x4_sob5_y_otsu.png'
    # save =  '/home/nakajima/work/Ecoli/data/new_nd049_s0_shape/lap_5_/processing_/bil3010x4_sob5_otsu_-y'
