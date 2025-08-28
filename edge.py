import cv2
from PIL import Image
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import copy
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball

from search_image import Search
from picture import Picture
from getshape import Getshape
import read_xml 
from skimage.morphology import skeletonize, thin
from skimage import data
from skimage.util import invert
import datetime
import setuptools
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import inspect

pic = Picture()
gsh = Getshape()
pic = Picture()

#実行した時の時間を取得
class folder:
    def __init__(self):      
        self.dt_now = datetime.datetime.now()

# 画像読み込み
def load_image(path):
    try:
        return cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    except:
        return tifffile.imread(path)

# 画像保存
def save_image(path,name, image):
    for i in range(len(name)):
        time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = path + name[i] + '_' + time + '.png'
        print(save_path)
        cv2.imwrite(save_path,image[i])

#スケルトン化 opencv   
def skelton(image):
    # 細線化(スケルトン化) THINNING_ZHANGSUEN
    skeleton1   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 細線化(スケルトン化) THINNING_GUOHALL 
    skeleton2   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    return skeleton1,skeleton2

#スケルトン化　skimage
def sk_skeleton(image):
    image1 = (image/ 255.).astype(np.float32)
    #normal
    skeleton1 = skeletonize(image1)
    #lee skl 3d用だから意味ないかも
    skeleton_lee = skeletonize(image, method='lee')
    #thin 
    thinned = thin(image1)
    #回数指定thin
    thinned_partial = thin(image1, max_num_iter=10)
    return skeleton1, skeleton_lee, thinned, thinned_partial



#以下testプログラム
#画像の表示
def plot_image(images):
    print(len(images))
    row = 1
    col = len(images)
    fig, ax = plt.subplots(nrows=row, ncols=col,figsize=(10,4))
    for i,img in enumerate(images):
        ax[i].imshow(img)
        # plt.imshow(images[i])
    plt.show() #なくても表示された。

# モルフォジー変換
def morphology_change(image):
    kernel = np.ones((3,3),np.uint8)
    mor_image = cv2.erode(image,kernel,iterations = 1)
    return mor_image

# 輪郭抽出　opencv
def find_outline(image):
    copyimage  = copy.deepcopy(image)
    result = np.zeros_like(copyimage)
    num =1
    while True:
        contours, hierarchy = cv2.findContours(copyimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        for group in contours:
            for i in group:
                result[i[0][1],i[0][0]] = num
                copyimage[i[0][1],i[0][0]] = 0                
        num +=1
        if not contours:
            break

    # result = cv2.drawContours(image, contours, -1, 125, 1)
    # 結果の表示
    # plt.imshow(result)
    # plt.show()
    return result

# 輪郭抽出
def new_find_outline(img):
    copyimg  = copy.deepcopy(img)
    num=1
    back_img = np.zeros_like(copyimg)
    binary_img = copyimg / 255
    copy_bin_img = copy.deepcopy(binary_img)

    reversed_binary_image = np.max(copy_bin_img) - copy_bin_img     
    shrink_image= ndimage.binary_dilation(reversed_binary_image)
    border_pixels = np.logical_and(shrink_image, copy_bin_img == 1)

    # plt.imshow(border_pixels)
    # plt.show()
    contour_image = np.where(border_pixels>0,255,0)
    return contour_image

 #輪郭線の探索,等高線の作成
def find_contour_line(img):
    copyimg  = copy.deepcopy(img)
    num=1
    back_img = np.zeros_like(copyimg)
    binary_img = copyimg / 255
    copy_bin_img = copy.deepcopy(binary_img)
    while True:
        reversed_binary_image = np.max(copy_bin_img) - copy_bin_img     
        shrink_image= ndimage.binary_dilation(reversed_binary_image)
        border_pixels = np.logical_and(shrink_image, copy_bin_img == 1)

        contour_image = np.where(border_pixels>0,num,0)
        back_img = contour_image + back_img
        copy_bin_img = copy_bin_img- border_pixels
        num +=1
        if np.all(copy_bin_img == 0):
            break
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    for i,xy in enumerate(centroids):
        copy_bin_img[int(xy[1]),int(xy[0])] = i+10
    print(f'len center : {np.max(labels)}')
    combin_img = labels+ copy_bin_img
    result = [img, back_img]
    return result

#中心点の作成
def search_center(img):
    img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    # 輪郭ごとの処理
    print(type(contours))
    print(print([len(v) for v in contours]))
    for i, contour in enumerate(contours):
        print(contour.shape)
        # 重心の計算
        m = cv2.moments(contour)
        if m['m00'] !=0:
            x,y= m['m10']/m['m00'] , m['m01']/m['m00']
            # print(f"Weight Center = ({x}, {y})")
            # 座標を四捨五入
            x, y = round(x), round(y)
            # 重心位置に x印を書く
            cv2.line(img_disp, (x,y), (x,y), (0, 0, 255), 1)
            cv2.line(img_disp, (x,y), (x,y), (0, 0, 255), 1)
        else:
            print('error : m00 = 0 ')
    print(i)
    # 結果の表示
    # plt.imshow(img_disp)
    # plt.show()

#独自のスケルトン作成
def create_skeleton(image):
    kernel = [[1,1,1],
              [1,0,1],
              [1,1,1]
            ]
    for i in range(0,10):
        image =0
def classification(image,save_path):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    print(n_labels)
    save_16bitimage(save_path,labels)


def save_16bitimage(path,image):
    if np.max(image)>255:
        # matrix_16bit = ((image / np.max(image)) * 65535).astype(np.uint16)
        matrix_16bit = (image).astype(np.uint16)
        cv2.imwrite(path, matrix_16bit)
    else:
        cv2.imwrite(path, matrix_16bit)

#メイン関数
def main():
    #全体画像
    path = '/home/nakajima/work/Ecoli/data'
    raw_path = os.path.join(path,'new_nd049_S0/c1/tiff/182_new.tiff') #16bit生画像
    save_path = os.path.join(path,'nd049_edge')#保存先

    input_dir = '/home/nakajima/work/Ecoli/data/nd049_label_image/'#正解ラベル画像のパス
    image_path = input_dir + '8bit_nd049_S1_C1_T0.tif' # 8bit Grayscale画像
    mask_path = input_dir + 'manual_label2/' + '8bit_nd049_S1_C1_T0_label.tif' # 生細胞マスク画像
    linemask_path = input_dir + 'line_label2/'+ '8bit_nd049_S1_C1_T0_binari.tiff' #線の正解ラベル画像 

    path = '/home/nakajima/work/Ecoli/code/Unet/result/20241029_010902_64_16_con/combined_image_or.png' #推論画像
    # path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label/image/8bit_nd049_S1_C1_T0.png' #新ラベル
    # path = '/home/nakajima/work/Ecoli/data/nd049_label_image/label/S1_C1_T0_elip_label.png #旧ラベル'
    # save_path = '/home/nakajima/work/Ecoli/code/seg_edge/test_image/'

    path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/test_manual/label/'
    input_path = path + '00059.tif'#分割の正解ラベル画像
    input_path = path + '00061.tif'#分割の正解ラベル画像
    mask = load_image(mask_path)
    a = (tifffile.imread(input_dir + 'manual_label2/8bit_nd049_S1_C1_T0_class2.tif'))
    
    # classification(mask,input_dir+'manual_label2/8bit_nd049_S1_C1_T0_class2.tif')
    # result1 = find_outline(mask)
    # result2 = new_find_outline(mask)
    # result3 = find_contour_line(mask)

    # mask_bg = np.array([mask,mask,mask])
    # mask_bg = mask_bg.transpose(1, 2, 0)
    # fitline(mask)
    # plot_image([result1,result2,result3[1]])
  
    # save = '/home/nakajima/work/Ecoli/code/seg_edge_length/prop_edge_length_image/'
    # cv2.imwrite(save+'contour_diff_overlay.png',dif)
    # norm_image = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX)
    # norm_image = cv2.applyColorMap(norm_image,cv2.COLORMAP_MAGMA)
    # cv2.imwrite('/home/nakajima/work/Ecoli/code/seg_edge_length/prop_edge_length_image/cv2cont_repeat_color.png',norm_image)

    # cv2.imwrite('/home/nakajima/Desktop/test3.png',img)
    # path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/test_manual/label/'
    # input_path = path + '00059.tif'
    # raw_image = load_image(input_path)
    # img3 = new_find_contour(raw_image)
    # norm_image = cv2.normalize(img1,None,0,255,cv2.NORM_MINMAX)
    # norm_image = cv2.applyColorMap(norm_image,cv2.COLORMAP_JET)
    # cv2.imwrite('/home/nakajima/work/Ecoli/code/seg_edge_length/prop_edge_length_image/findcontours.png',norm_image)



if __name__ == "__main__": 
    main() 


# 輪郭情報の取得
# contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

# # 画像表示用に入力画像をカラーデータに変換する
# # img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # 輪郭ごとの処理
# for i, contour in enumerate(contours):
#     # 輪郭の面積を求める
#     area = cv2.contourArea(contour, True)
#     print(f"面積[{i}]: {area}")

#     # 輪郭座標
#     for point in contour:
#         print(point[0])

# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
