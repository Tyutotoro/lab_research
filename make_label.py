import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

import tifffile as tiff
import os
from scipy import ndimage
import copy
import inspect
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.ndimage import center_of_mass


from skimage import filters, morphology, measure
from scipy.spatial import ConvexHull
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from getshape import Getshape
from picture import Picture 
pic = Picture()
gsh = Getshape()


#画像読み込み
def load_image(path):
    try:
        input_image = tiff.imread(path)
    except:
        input_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return input_image

#classfication
def load_images_and_labels(path, save_path):
    label_folder = os.path.join(path, "label")
    n, label_list = pic.get_picture(label_folder)
    os.makedirs(os.path.join(save_path, 'label'), exist_ok=True)

    for i, file_name in enumerate(label_list):
        label_path = os.path.join(label_folder, file_name)
        # print(label_path)
        # 生画像とラベル画像を読み込み
        label = load_image(label_path)
        img = np.zeros(label.shape)
        n_labels, class_label, stats, centroids = cv2.connectedComponentsWithStats(label)

        class_label_save = class_label.astype('uint8')
        
        tiff.imwrite(os.path.join(save_path, 'label',f'{i + 1:05d}.tif'), class_label_save)

#大腸菌ごとのクラスラベル(値入り)の作成
def make_class_label(image):
    # class_image = cv2.findcontour(image)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    print(n_labels)
    print(labels)
    print(centroids)
    plt.imshow(labels)
    plt.show()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            # print(labels[i,j])
            if labels[i,j] > 0:
                plt.text(j, i, str(labels[i, j]), 
                    ha='center', va='center', color='black', fontsize=5)
        print(i)
    plt.imshow(labels)
    plt.show()
    return image

def over(rpath, lpath):
    raw = load_image(rpath)
    label = load_image(lpath)
    back = np.zeros(raw.shape)
    label_color = np.array([back,back,label])
    label_color = label_color.transpose(1, 2, 0)
    label_color = label_color.astype('uint8')
    raw_color = np.array([raw,raw,raw])
    raw_color = raw_color.transpose(1, 2, 0)

    image = gsh.overlay_img(raw_img=raw_color, proce_img=label_color, num=[1, 0.1 ,0])
    plt.imshow(image)
    plt.show()
    return image

def main():
    input_dir = '/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16'
    # input_path = input_dir + 'manual_label/image/' + '8bit_nd049_S1_C1_T0.png' 
    # label_path = input_dir +'manual_label2/foreground.tif'
    # raw_path = input_dir + '8bit_nd049_S1_C1_T0.tif'
    save_path = '/home/nakajima/work/Ecoli/code/stardist/scaledata/256_16'
    save_list = ['test_manual2_class']
    label_list = ['test_manual2']
    # image = over(raw_path,label_path)
    for label, save in zip(label_list, save_list):
        inp_dir = os.path.join(input_dir, label)
        sa_path = os.path.join(save_path, save)
        load_images_and_labels(inp_dir, sa_path)
    


if __name__ == "__main__":
    main()