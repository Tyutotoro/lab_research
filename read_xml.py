import cv2
from PIL import Image
import tifffile
import numpy as np
import matplotlib.pyplot as pt
import os
import glob
from copy import copy
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
import xml.etree.ElementTree as ET
import csv
from search_image import Search
from picture import Picture
from getshape import Getshape
import pandas as pd
import matplotlib.pyplot as plt
pic = Picture()
gsh = Getshape()

def read_xml(xml):
    tree = ET.parse(xml)
    #一番上の階層の要素を取り出します
    root = tree.getroot()
    position = []
    # ルートノードの表示
    # 子ノードを読み込む
    for child1 in root:
        for child2 in child1:
            for child3 in child2:
                if child3.tag == 'Region': 
                    for child4 in child3:
                        for child5 in child4:
                            position.append([float(child5.attrib['X']),float(child5.attrib['Y'])])
    position = np.array(position)
    return position

def read_xml_length(xml):
    tree = ET.parse(xml)
    #一番上の階層の要素を取り出します
    root = tree.getroot()
    position = []
    # ルートノードの表示
    # 子ノードを読み込む
    for child1 in root:
        for child2 in child1:
            for child3 in child2:
                if child3.tag == 'Region': 
                    position.append([int(child3.attrib["Id"]),float(child3.attrib["Length"])])
    position = np.array(position)
    return position    

def cal_deg(vec_a,vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        cos_theta = dot_product / (norm_a * norm_b)
        # print(cos_theta)
        theta = np.arccos(cos_theta)
        # 弧度法から度に変換
        deg = np.degrees(theta)
        return deg

#楕円のラベル             
def make_label_circle(img,position):
    for i in range(0,len(position)-1,2):
        L = 2
        n = 0
        a = position[i][0]-n
        b = position[i][1]-n
        c = position[i+1][0]-n
        d = position[i+1][1]-n
        lengh = np.sqrt((c-a)**2+(d-b)**2)
        mid = [(a+c)/2,(b+d)/2]
        #ベクトル
        vec_a = np.array([mid[0] - a, mid[1] - b])
        vec_b = np.array([0,1])
        deg = cal_deg(vec_a, vec_b)
        # print((d-b)/(c-a))
        if ((d-b)/(c-a))>0:
            deg = -1*deg
        cv2.ellipse(img, (mid,[L, lengh] ,deg), color =255, thickness=-1,)
        posi_c = position[i+1][0].astype(np.int64)
        posi_d = position[i+1][1].astype(np.int64)
    return img

#線のラベル
def make_label_line(img,position):
    for i in range(0,len(position)-1,2):
        # cv2.line(img,(int(position[i][0]),int(position[i][1])),(int(position[i+1][0]),int(position[i+1][1])),color =i/2, thickness = 1)
        cv2.line(img,(int(position[i][0]),int(position[i][1])),(int(position[i+1][0]),int(position[i+1][1])),color =2, thickness = 1)
    return img


def make_label_porigonc_ircle(position):
    #円と四角のラベル
        # 線分の方向ベクトルを計算
        # if c- a == 0:
        #     unit_vector = np.array([1, 0])
        #     posi_1 = (np.array([a, b]) - L * unit_vector).astype(np.int64)
        #     posi_2 = (np.array([a, b]) + L * unit_vector).astype(np.int64)
        #     posi_3 = (np.array([c, d]) + L * unit_vector).astype(np.int64)
        #     posi_4 = (np.array([c, d]) + L * unit_vector).astype(np.int64)
        # else:
        #     # m = (d- b) / (c-a)
        #     # unit_vector = np.array([1, m]) / np.sqrt(1 + m**2)
        #     len1 = (c - a)**2
        #     len2 = (d - b)**2
        #     vec = (np.sqrt(L/(len1+len2)))*((d - b)/(c - a))
        #     print(vec)
        #     vec_posi = np.array([vec,vec])
        #     x1y1 = position[i]-vec_posi
        #     x2y2 = position[i]+vec_posi
        #     x3y3 = position[i+1]+vec_posi
        #     x4y4 = position[i+1]-vec_posi
        #     posi_1 = x1y1.astype(np.int64)
        #     posi_2 = x2y2.astype(np.int64)
        #     if (x3y3[1]-x2y2[1])/(x3y3[0]-x2y2[0]) == (d-b)/(c-a):
        #         posi_3 = x3y3.astype(np.int64)
        #         posi_4 = x4y4.astype(np.int64)
        #     else:
        #         posi_3 = x4y4.astype(np.int64)
        #         posi_4 = x3y3.astype(np.int64)
        #         print("else")

        # posi_a = position[i][0].astype(np.int64)
        # posi_b = position[i][1].astype(np.int64)
        # posi_c = position[i+1][0].astype(np.int64)
        # posi_d = position[i+1][1].astype(np.int64)

        # cv2.fillConvexPoly(img, np.array([posi_1,posi_2,posi_3,posi_4]), (255))
        # cv2.rectangle(black_img,x1y1_int,x2y2_int,(255),thickness= -1)
        # cv2.circle(black_img, (posi_a,posi_b), L, (255), thickness=-1)
        # cv2.circle(black_img, (posi_c,posi_d), L, (255), thickness=-1)
    return 0


def make_label(xml,raw_img,save_path):
    #xmlデータを読み込みます
    img = tifffile.imread(raw_img)
    position = read_xml(xml)
    black_img = np.zeros((1024,7117))

    # black_img = make_label_circle(black_img, position)
    result = make_label_line(black_img,position)
    #カラー化
    # result = cv2.applyColorMap(result,cv2.COLORMAP_INFERNO)#COLORMAP_HOT
    
    # cv2.imwrite(f'{save_path}/S1_C1_T0_line_color_label.tiff',result)
    # cv2.imwrite(f'{save_path}/S1_C1_T0_line_color_label.png',result)

    result_int = result.astype(np.uint8)
    over = gsh.overlay_img(img, result_int,[0.8,0.2,2])
    # cv2.imwrite(save_path+'/S0_C1_T0_line_color_overlay.png',over)
    print('maked_label')

def main():
    xml = "/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S1_C1_T0.xml"
    raw_img = "/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S1_C1_T0.tif"
    save_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/line_label"
    # make_label(xml,raw_img,save_path)
    # lengths = read_xml_length(xml)
    result = make_label_line(xml)
    plt.savefig(save_path+'/nd049_S1_C1_T0_line.png')
    # with open(save_path+'/nd049_S1_C1_T0_line.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(lengths)
    # make_label(xml,raw_img,save_path)



if __name__ == "__main__":
    main()