from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tifffile
import os
import glob
from search_image import Search
from copy import copy


dist1=Search()

class Picture:
    def get_directories(self,path):
        directories_1 = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # directories_2 = {d: [sd for sd in os.listdir(os.path.join(path, d)) if os.path.isdir(os.path.join(path, d, sd))]
        #                 for d in directories_1}
        print("directories_1",directories_1)
        # print("directories_2",directories_2)
        # return directories_1, directories_2
        return directories_1

    
    #path下の画像数と画像のリストの取得
    def get_picture(self,path):
        path = path + '/'
        if glob.glob(path+'*.tiff'):
            name = glob.glob(path + '*.tiff')
        elif glob.glob(path+'*.png'):
            name = glob.glob(path +'*.png')
        elif glob.glob(path+'*.tif'):
            name = glob.glob(path+'*.tif')
        else:
            print("error")
            name = ''

        num=len(name)
        filename=dist1.sort_file(name)
        return num, filename       

