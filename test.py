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
from search_image import Search
from picture import Picture
from getshape import Getshape
import read_xml 
import re
import shutil
import random

pic = Picture()
gsh = Getshape()
dist1 = Search()
random.seed(0)

##画像のトリミング&分割
def trim_and_split_image(input_path, output_path, trim_left=94, trim_right=95, target_size = None, overlap =False, oversize= None):
    # 102,103
    # 画像を読み込む
    try:
        input_image = tifffile.imread(input_path)
    except:
        input_image = cv2.imread(input_path,cv2.IMREAD_GRAYSCALE)
    os.makedirs(output_path,exist_ok=True)
    print(input_path, output_path,target_size)
    # 画像をトリミング
    width = input_image.shape[1] - trim_right
    height = input_image.shape[0]
    trimmed_image = input_image[0 : height, trim_left : width]
    # トリミングした画像を均等に分割
    if overlap:
        slice_image = over_split(trimmed_image, target_size, oversize)
        print('over_split')
    else:
        slice_image = gsh.split(trimmed_image,sizeh=target_size, sizev=target_size)
    # 分割した画像を保存
    slice_image_np  = np.array(slice_image)
    
    print(len(slice_image))
    num = get_max_tile_id(output_path)
    if num == -1:
        num = 0
    for i in range(len(slice_image)):      
        image = slice_image_np[i, :, :]
        output_filepath = os.path.join(output_path, f"{i + num + 1:05d}.tif")
        cv2.imwrite(output_filepath, image)

def over_split(image, tile_size, overlap):
    # 画像を読み
    height, width= image.shape
    print(height,width)
    tiles = []
    y_steps = range(0, height - overlap, tile_size - overlap)
    x_steps = range(0, width - overlap, tile_size - overlap)
    
    # 画像を分割する
    for y in y_steps:
        for x in x_steps:
            # タイルのサイズが画像の端を超える場合に、端の部分を重複させる
            tile_y_end = min(y + tile_size, height)
            tile_x_end = min(x + tile_size, width)
            # print(f'y:y,x:x {y,tile_y_end, x,tile_x_end}')
            tile = image[y:tile_y_end, x:tile_x_end]

            # 縦または横が足りない場合、重複部分を増やしてタイルサイズを合わせる
            if tile_y_end - y < tile_size or tile_x_end - x < tile_size:
                print('if')
                y_overlap = tile_size - (tile_y_end - y)
                x_overlap = tile_size - (tile_x_end - x)
                tile = image[max(0, y - y_overlap):tile_y_end, max(0, x - x_overlap):tile_x_end]

            tiles.append(tile)
    return tiles

#RGB画像の二値化
def binarize_image(input_image_path):
    # 画像をグレースケールで読み込む
    im = cv2.imread(input_image_path)
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # img_gray = 0.8 * im[:, :, 2] + 0.1 * im[:, :, 1] + 0.1 * im[:, :, 0]
    # img_gray = img_gray.astype(np.uint8)
    # 2値化処理
    binary_image = gsh.Otsumethed(img_gray)
    binary_image = cv2.bitwise_not(binary_image)
    # 画像保存
    return binary_image

def assign_values_to_white_regions(binary_image):
    input_image = tifffile.imread(binary_image)
    # 白い領域のラベリングを行う
    labeled_image = measure.label(input_image, connectivity=2)
    # ユニークなラベル（白い領域の数）を取得
    unique_labels = np.unique(labeled_image)
    # 白い領域の数を取得
    num_white_regions = len(unique_labels) - 1  # 背景のラベル0を除く
    # 画像の値の範囲を0からnまでに変更
    for label in range(1, num_white_regions + 1):
        labeled_image[labeled_image == label] = label

    return labeled_image

def change_intensity_range(image, new_min, new_max):
    old_min = np.min(image)
    old_max = np.max(image)
    stretched = (image - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    return np.clip(stretched, new_min, new_max).astype(np.uint8)


def rename_images(folder_path,save_folder_path):
    # フォルダ内の画像ファイルを取得
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tiff', '.tif', 'png'))]
    
    for i, old_name in enumerate(sorted(image_files)):
        # 拡張子を取得
        _, extension = os.path.splitext(old_name)
        # 新しいファイル名を生成 
        new_name = f"{i + 1:05d}{extension}"
        old_path = os.path.join(folder_path, old_name)
        print(old_path)
        new_path = os.path.join(save_folder_path, new_name)
        # ファイルをリネーム
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")


def IoU(pred, target):
    num_classes = 2
    ious = [0]*(num_classes+1)
    array_ious = list()
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            array_ious.append(float('nan'))  # If there is no ground truth, IoU is undefined
            ious[cls] = array_ious[cls]*100#クラスclsのIoU
        else:
            array_ious.append(intersection / union)
            ious[cls] = array_ious[cls]*100#クラスclsのIoU
    ious[num_classes] = (array_ious[1])
    # ious[num_classes] = (array_ious[1]+array_ious[2])/2
    # ious[num_classes+1] = statistics.mean(array_ious)#全クラスのmIoU
    ious[num_classes] = statistics.mean(array_ious)*100#全クラスのmIoU
    return ious

def binarize_and_save_images(folder_path, output_folder):
    # フォルダ内のすべてのファイルを取得
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        # ファイルの拡張子が画像形式の場合のみ処理
        if file_name.lower().endswith(('.tiff', '.tif')):
            file_path = os.path.join(folder_path, file_name)
            image = tifffile.imread(file_path)

            if image is not None:
                # 2値化処理（0以外の値を255に変換）
                _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

                # 出力先ファイルパスを構築
                output_path = os.path.join(output_folder, file_name)

                # 画像を保存
                cv2.imwrite(output_path, binary_image)

def seg(im):
    contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(len(contours)):
    #     if cv2.contourArea(contours[i]) > (im.shape[0] * im.shape[1]) * 0.005:
    #         # img_with_area = deepcopy(img_BGR)
    #         im = cv2.fillPoly(im, [contours[i][255,255,255]], 0, lineType=cv2.LINE_8, shift=0)
    for i in range(len(contours)):
        cnt = contours[i]
        im = cv2.drawContours(im, [cnt], 0, 255, -1)
    return im

def RGB_label(path, save_path):
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.tiff', '.tif'))]
    for i, old_name in enumerate(sorted(image_files)):
        image_path = os.path.join(path, old_name)
        binary_image=binarize_image(image_path)
        binary_image = seg(binary_image)
        output_path = os.path.join(save_path, f"smgray{i-1 + 1:03d}.tif")
        cv2.imwrite(output_path, binary_image)
    return 0


def get_max_tile_id(output_dir, prefix='', suffix='.tif'):
    files = os.listdir(output_dir)
    
    pattern = re.compile(rf'{prefix}(\d+){suffix}')
    ids = [
        int(pattern.match(file).group(1))
        for file in files if pattern.match(file)
    ]
    return max(ids, default=-1)


def multi_image_split(input_path,output_path,trim=None,target_size=None,overlap = False,oversize = None):
    raw_image_list = ['8bit_nd049_S0_C1_T0.tiff','8bit_nd049_S0_C1_T1.tiff','8bit_nd049_S1_C1_T0.tif']
    raw_label_list = ['8bit_nd049_S0_C1_T0.png','8bit_nd049_S0_C1_T1.png','8bit_nd049_S1_C1_T0.png'] 
    save_list = ['train_manual', 'train_manual', 'test_manual_pre']
    if overlap:
        output_path = os.path.join(output_path,str(target_size)+'_'+str(oversize))
    else:
        output_path =   os.path.join(output_path,str(target_size))
    for i in range(len(raw_image_list)):
        image_path = os.path.join(input_path, raw_image_list[i])
        label_path = os.path.join(input_path, raw_label_list[i])
        # trim_and_split_image(image_path, os.path.join(output_path,save_list[i],'image'), trim_left=trim[0], trim_right=trim[1], target_size=target_size, overlap=overlap,oversize=oversize)
        trim_and_split_image(label_path, os.path.join(output_path,save_list[i],'label'), trim_left=trim[0], trim_right=trim[1], target_size=target_size, overlap=overlap,oversize=oversize)


def get_images(path):
    print("get_images")
    """
    指定されたパスのtrainとvalディレクトリの中身を取得する。
    """
    train_images = []
    label_images = []
    
    train_path = os.path.join(path, 'image')
    print(train_path)
    if os.path.exists(train_path):
        t_num, train_images = pic.get_picture(train_path)
        print(t_num)

    label_path = os.path.join(path, 'label')
    if os.path.exists(label_path):
        v_num, label_images = pic.get_picture(label_path)
        print(v_num)
    
    return train_images, label_images

def remove_zero_pixel_images(image_list):
    """
    valディレクトリの中身のうち、全ての画素の値が0である画像をリストから削除する。
    """
    print("remove")
    valid_images = []
    removed_images = []
    for image_name in image_list:
        try:
            img = tifffile.imread(image_name)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img) 
            if (min_val, max_val) != (0, 0):
                valid_images.append(image_name)
            else:
                removed_images.append(os.path.basename(image_name))
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
    
    return valid_images, removed_images

def copy_images(src_path, dst_path, image_list, subdir):
    """
    画像を指定されたディレクトリ構造でコピーする。
    """
    dst_subdir_path = os.path.join(dst_path, subdir)
    if not os.path.exists(dst_subdir_path):
        os.makedirs(dst_subdir_path)
    print(dst_subdir_path)
    for image_name in image_list:
        image_name = os.path.basename(image_name)
        src_image_path = os.path.join(src_path, subdir, image_name)
        dst_image_path = os.path.join(dst_subdir_path, image_name)
        shutil.copyfile(src_image_path, dst_image_path)

def move_images(image_list, dst_path, suddir):
    for image_name in image_list:
        shutil.move(image_name, os.path.join(dst_path,suddir))


def remove_no_ecoli_image(input_path, output_path):
    directories_1, directories_2 = pic.get_directories(input_path)
    
    for dir1 in directories_1:
        for dir2 in directories_2[dir1]:
            dir2_path = os.path.join(input_path, dir1, dir2)
            if dir2 != "val":
                train_images, val_images = get_images(dir2_path)
                valid_val_images, removed_images = remove_zero_pixel_images(val_images)
                # 削除された画像と同じ名前をtrainのリストからも削除
                valid_train_images = [img for img in train_images if os.path.basename(img) not in removed_images]
            else:
                print("valです")
                valid_train_images = []
                valid_val_images = []
            # 新しい構造のディレクトリを作成
            output_dir2_path = os.path.join(output_path, dir1, dir2)
            os.makedirs(os.path.join(output_dir2_path, 'image'), exist_ok=True)
            os.makedirs(os.path.join(output_dir2_path, 'label'), exist_ok=True)
            
            # 残った画像を新しいディレクトリにコピー
            copy_images(dir2_path, output_dir2_path, valid_train_images, 'image')
            copy_images(dir2_path, output_dir2_path, valid_val_images, 'label')         
                
    print("Processing complete. Check the output directory for results.")
            
def move_val_image(input_path):
    directories_1 = pic.get_directories(input_path)
    
    for dir1 in directories_1:
        # for dir2 in directories_2[dir1]:
        dir2_path = os.path.join(input_path, dir1)
        print(f'dir: {dir1}')
        if dir1 == "train_manual" :
            img_images, label_images = get_images(dir2_path)
            train_path = dir2_path
            print(train_path)
        if dir1 == "val_manual":
            val_path = dir2_path
            print(val_path)
        else:
            print("test")
        # 画像数の10%を移動
    num_images_to_move = random.sample(img_images,int(len(img_images)*0.1)) 
    labels_to_move = [s.replace('image', 'label') for s in num_images_to_move]
    if not os.listdir(val_path + '/image'):
        move_images(num_images_to_move, val_path, 'image')
    if not os.listdir(val_path + '/label'):
        move_images(labels_to_move, val_path, 'label')
        print("Processing complete. Check the output directory for results.")           

def move_lstmunet_image(path,val_path):
    directories1,directories2 = pic.get_directories(path)

    for dir1 in directories1:
        for dir2 in directories2[dir1]:
            dir1_path = os.path.join(path, dir1)
            if dir2 == "image":
                img_images, label_images = get_images(dir1_path)
                save_path = dir1_path
            if dir2 == "label":
                lavel_path = os.path.join(dir1_path,dir2)
                

    # 画像数の10%を移動
    image_to_move = random.sample(img_images,len(img_images)// 20)
    label_to_move = [os.path.join(lavel_path, os.path.basename(filename)) for filename in image_to_move]
    # 画像を新しいディレクトリにコピー
    move_images(image_to_move, save_path, '03')
    move_images(label_to_move, save_path, '03_GT')


def make_labelset(input_path, save_path,targetsize=64,oversize=16):
    for i in len(input_path):
        os.makedirs(save_path[i], exist_ok=True)
        trim_and_split_image(input_path=input_path[1],output_path=save_path[1],target_size=targetsize,overlap=True,oversize=oversize)


def main():
    # renameの実行
    # c = 'color_mask'
    # a  = '/home/nakajima/work/Ecoli/code/cellpose/64_16_result/'
    # a = os.path.join(a,c)
    # b  = '/home/nakajima/work/Ecoli/code/cellpose/64_16_result/'
    # c = 'rename_'+c 
    # b = os.path.join(b,c)
    # rename_images(a,b)

    # multi_image_split(path,save_path,size,over_switch,over)
    #256画像作成  
    # input_path = '/home/nakajima/work/Ecoli/data/nd049_256/S1_C1'
    # directories_1 = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    # filename = dist1.sort_file(directories_1)
    # for num in range(0,146):    
    #     out_path = '/home/nakajima/work/Ecoli/data/nd049_256/S1_C1_timelapse/' + str(f'{num:03d}')
    #     os.makedirs(out_path,exist_ok=True)
    #     for path in enumerate(filename):
    #         inp_path = os.path.join(input_path,path[1])
    #         _,file1 = pic.get_picture(inp_path)
    #         # print(file1[46])
    #         output_path = os.path.join(out_path, str(f'T{path[0]:04d}.tif'))
    #         shutil.copyfile(file1[num-1], output_path)

    # input_path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/val_manual'
    # input_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_label.tif'
    input_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S3_C1_T0.tif'
    save_path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/acrr/image'
    # save_path = '/home/nakajima/work/Ecoli/code/Unet/removed_data/64_16/val_manual'
    trim_and_split_image(input_path=input_path, output_path=save_path,trim_left=94,trim_right=95,target_size=64,overlap=True,oversize=16)
    # make_labelset(input_path=input_path, save_path=save_path)
    # multi_image_split(input_path=input_path,output_path=save_path,trim=[94,95],target_size=64,overlap=True,oversize=16)
    # move_val_image(save_path)

if __name__ == "__main__":
    main() 
   