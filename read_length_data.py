import os, sys, random, math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def convert_img(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
        #     print(img[i][j])
            if img[i][j] > 0:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img
                 
def match_label(txtfilee,csvfile):
    return 0

def read_file(csvpath,txtpath ):
    total_list =[]
    total_list.append(['name','length','pix_length','x1','y1','x2','y2'])
    print(os.path.splitext(csvpath))
    df = pd.read_csv(csvpath)
    lengths = df[['Label','Length']]       
    print(lengths)
    with open(txtpath, 'r') as txtfile:
        while True:
            # 1行目（name）の読み込み
            name_line = txtfile.readline().strip()
            if not name_line:  # 空行またはファイルの終わり
                break
            # 2行目（座標1）の読み込み
            coord1_line = txtfile.readline().strip()
            c1_1 =int(coord1_line.split(',')[0])
            c1_2 =int(coord1_line.split(',')[1])
            # 3行目（座標2）の読み込み
            coord2_line = txtfile.readline().strip()
            c2_1 = int(coord2_line.split(',')[0])
            c2_2 = int(coord2_line.split(',')[1])
            # 4行目の空行を読み飛ばす
            txtfile.readline()
            # CSVのlabelと一致するか判定し、一致した場合にlengthを抽出
            for i in range(len(lengths)):
                name = name_line.split()[1].split(':')[0]
                if name in lengths.iloc[i,0] :
                    # print(name_line)
                    length = lengths.iloc[i,1]
                    pix_length = length*3.523
                # 抽出したlengthと座標を1つの配列にまとめ、リストに追加
                    total_list.append([name, length,pix_length, c1_1,c1_2,c2_1,c2_2 ])
    print(total_list)
    return total_list

def write_csv(path,data):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def main():    
    path = '/home/nakajima/work/Ecoli/data/nd049_label_image/line_label3/'
    img = cv2.imread(path + '8bit_nd049_S1_C1_T0_contour.tiff', cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    img = convert_img(img)
    cv2.imwrite(path + '8bit_nd049_S1_C1_T0_binari.tiff',img)
    cv2.imwrite(path + '8bit_nd049_S1_C1_T0_binari.png',img)
    csv = path + '8bit_nd049_S1_C1_T0_linelabel.csv'
    txt = path + '8bit_nd049_S1_C1_T0.txt'
    savepath = path + '8bit_nd049_S1_C1_T0_length.csv'
    # label = read_file(csv,txt)
    # write_csv(savepath,label)

if __name__ == "__main__":
    main() 
   

   #0586-6341