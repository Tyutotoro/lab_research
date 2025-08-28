from csbdeep.utils import normalize
# from stardist import random_label_cmap
from stardist.stardist.models.model2d import Config2D, StarDist2D
from stardist.stardist.data.images import test_image_nuclei_2d
from stardist.stardist.plot.render import render_label

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow_datasets as tfds
import shutil 
import stackview
import matplotlib.pyplot as plt
from skimage.data import human_mitosis
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import statistics
import warnings
# from keras.preprocessing.image import load_img

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tifffile as tiff
import cv2
import inspect
import pandas as pd
from stardist.matching import matching
import csv
import re
import glob


class Search:
    #file読み込み
    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text) ]
    
    def sort_file(self,file):
        return sorted(file, key=self.natural_keys)
    
    def seq_file(self,file):
        source_file, target_file=[],[]
        files=(self.sort_file(file))
        num=len(files)
        for i in range(num):
            if '0_' in files[i]:
                source_file.append(files[i])
            elif 'src' in files[i]:
                source_file.append(files[i])
            else :
                target_file.append(files[i]) 
        return source_file, target_file

dist1 = Search()
    #path下の画像数と画像のリストの取得
def get_picture(path):
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


#画像読み込み
def load_image(path):
    try:
        input_image = tiff.imread(path)
    except:
        input_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return input_image

# def stardist(imagepath,savepath):    
#     num,image_list = get_picture(imagepath)
#     print(image_list[0])
#     model = StarDist2D.from_pretrained('2D_versatile_fluo')
#     # print((model.logdir))
#     for i, image_name in enumerate(image_list):
#         image = tiff.imread(image_name)
#         # print(image)
#         # stackview.insight(image)

#         # image2 = normalize(image, pmin=1, pmax=99.8)

#         # パーセンタイル値の計算
#         lower_percentile = np.percentile(image, 1)   # 下位1パーセンタイル
#         upper_percentile = np.percentile(image, 99.8)  # 上位99.8パーセンタイル

#         # 画像データの正規化
#         image2 = (image.astype("int16") - lower_percentile) / (upper_percentile - lower_percentile)

#         labels, details = model.predict_instances(image2)

#         tiff.imwrite(savepath + '/'+ format(i, '05d')  + '.tif',labels)
#         labels_img = PIL.Image.fromarray(labels)
#         labels_img.save(savepath + '_png/'+ format(i, '05d')  + '.png')

#         # plt.figure(figsize=(5,5))
#         # plt.imshow(image2, clim=(0,1), cmap='gray')
#         # plt.imshow(labels, cmap=random_label_cmap(), alpha=0.5)
#         # plt.axis('off')

def load_images_and_labels(data_type_folder,cla = False,ave = 99.113,std = 7.14):
    image_folder = os.path.join(data_type_folder, "image")
    n, image_list = get_picture(image_folder)
    label_folder = os.path.join(data_type_folder, "label")
    n, label_list = get_picture(label_folder)

    images = []
    labels = []
    for i in range(len(image_list)):
        image_name = image_list[i]
        label_name = label_list[i]
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, label_name)

        # 生画像とラベル画像を読み込み
        img = load_image(image_path)
        # img = (img-ave)/std
        class_label = load_image(label_path)
        if cla == True:
            n_labels, class_label, stats, centroids = cv2.connectedComponentsWithStats(class_label)

        images.append(img)
        labels.append(class_label)
    return images, labels

def parse_and_write_metrics(file_path, metrics_str):
    # メトリクス文字列からキーと値を抽出
    pattern = r'(\w+)=([-+]?\d*\.?\d+)'  # キー=値 の形式を正規表現で抽出
    parsed_metrics = re.findall(pattern, metrics_str)

    # キーと数値を分離
    keys = [key for key, _ in parsed_metrics]
    values = [float(value) if '.' in value else int(value) for _, value in parsed_metrics]

    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(keys)
        writer.writerow(values)

def acc_csv(path,name):
    input_file =  path + '/'+ name 
    output_file = path + "/non_0_" + name
    df = pd.read_csv(input_file)

    # n_pred が非0の行をフィルタリング
    filtered_df = df[df['n_pred'] != 0].copy()
    row_indices = filtered_df.index.tolist()
    filtered_df.insert(0, "extracted_rows", row_indices)

    mean_values = filtered_df.mean(axis=0)
    mean_values["extracted_columns"] = "mean_values"
    empty_row = {col: "" for col in filtered_df.columns}
    filtered_df = pd.concat([filtered_df, pd.DataFrame([empty_row]), pd.DataFrame([mean_values])], ignore_index=True)

    # 新しいCSVファイルに保存
    filtered_df.to_csv(output_file, index=False)

def f1_iou(pred,target, num_classes):
        warnings.simplefilter('ignore')
        # print(num_classes)
        if num_classes < 2:
            num_classes = 2
        ious = [0]*(num_classes+2)
        no_nan_ious = [0]*(num_classes+2)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        array_ious = []
        anyious =0
        # print(num_classes, len(ious))

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = np.sum(pred_inds & target_inds)
            union = np.sum(pred_inds | target_inds)
            
            if union == 0:
                # array_ious.append(float('nan'))  # If there is no ground truth, IoU is undefined
                array_ious.append(np.nan)  # If there is no ground truth, IoU is undefined
                ious[cls] = array_ious[cls]*100#クラスclsのIoU
            else:
                array_ious.append(intersection / union)
                ious[cls] = array_ious[cls]*100#クラスclsのIoU
        # if num_classes == 2:
        #     ious[num_classes] = (array_ious[1])
        #     ious[num_classes] = statistics.mean(array_ious)*100#全クラスのmIoU
        # else:
        for i in range(num_classes-1):
                anyious += array_ious[i+1]
        anyious = anyious/(num_classes-1)
        ious[num_classes] = anyious
        ious[num_classes+1] = statistics.mean(array_ious)*100#全クラスのmIoU
        valid_ious = [iou for iou in array_ious if not np.isnan(iou)]
        no_nan_ious[num_classes] = statistics.mean(valid_ious) * 100 if valid_ious else float('nan') 
        
        #f1スコアの計算
        # pred = pred.cpu().detach().numpy()
        # target = target.cpu().detach().numpy()  
        # f1score = [recall_score(target,pred)*100,
        #         precision_score(target,pred)*100,
        #         f1_score(target,pred)*100]
        return ious, no_nan_ious
            

def save_score(score,save_path):
    file_exists = os.path.exists(save_path)

    with open(save_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(score)

def training_stardist(image_path,save_path,folder_list):
    model_pretrained = StarDist2D.from_pretrained('models/examples/2D_demo')
    # shutil.copytree(model_pretrained.logdir,save_path, dirs_exist_ok=True)
    name = 'metrics4.csv'
    batchsize = 5
    epochs = 400
    config = Config2D(
    # change some training params 
    train_batch_size = batchsize,
    train_learning_rate = 1e-4,
    train_epochs = epochs,
    train_patch_size = (256,256),
    # grid = (1,1) 
    )
    model = StarDist2D(config, name=None, basedir=save_path)

    train_dir =  os.path.join(image_path, folder_list[0])
    val_dir = os.path.join(image_path, folder_list[1])
    test_dir = os.path.join(image_path, folder_list[2])

    x_train, y_train = load_images_and_labels(train_dir,False)
    x_val, y_val = load_images_and_labels(val_dir,False)
    x_test, y_test = load_images_and_labels(test_dir,True)

    img = test_image_nuclei_2d()
    print(type(img))
    print(img.shape)
    print(img.dtype)
    plt.imshow(img)
    plt.show()
    print(img)
    # print(model)
    print(model_pretrained)
    labels, _ = model.predict_instances(normalize(img))

    print(labels)
    print(type(labels))
    print(labels.shape)
    print(labels.dtype)
    
    plt.subplot(1,3,1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("input image")

    plt.subplot(1,3,2)
    plt.imshow(render_label(labels, img=img))
    plt.axis("off")
    plt.title("prediction + input overlay")

    plt.subplot(1,3,3)
    plt.imshow(labels)
    plt.axis("off")
    plt.title("input image")

    plt.show()
    # model.config.use_gpu = True
#train
    # history = model.train(X = x_train,Y = y_train,  validation_data=(x_val,y_val),seed = 3407)
    # df = pd.DataFrame(history.history)
    # df.to_csv(os.path.join(save_path , 'history2.csv'))
    # plt.figure(10,6)
    # plt.plot(range(1, epochs+1), history.history['loss'], label="training loss")
    # plt.plot(range(1, epochs+1), history.history['val_loss'], label="validation loss")
    # plt.xlabel('Epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.savefig(os.path.join(save_path, 'loss2.png'), format="png")
    # plt.figure(10,6)
    # plt.plot(range(1, epochs+1), history.history['dist_dist_iou_metric'], label="dist_dist_iou_metric")
    # plt.plot(range(1, epochs+1), history.history['val_dist_dist_iou_metric'], label="val_dist_dist_iou_metric")
    # plt.xlabel('Epochs')
    # plt.ylabel('iou')
    # plt.legend()
    # plt.savefig(os.path.join(save_path, 'iou2.png'), format="png")
    # model.keras_model.save(os.path.join(save_path,"stardist_trained_model.h5"))

    # 予測
    save_pred_path = os.path.join(save_path,'pred_image')
    save_pred_nor_path = os.path.join(save_path,'pred_normalize_image')
    os.makedirs(save_pred_path,exist_ok=True)
    os.makedirs(save_pred_nor_path,exist_ok=True)
    for i, test_img in enumerate(x_test):
        instance = model_pretrained.predict_instances(
            img = test_img,  # 入力画像のnumpy配列
            axes = None,  # 画像の軸情報がない場合、Noneを指定
            normalizer = True,  # 正規化
            n_tiles = None,  #分割するタイル数の設定
            show_tile_progress = True  # 進捗を表示する場合はTrueを指定
        )
        iou, non_nan_iou =  f1_iou(instance[0],y_test[i],len(np.unique(y_test[i])))
        _, semantic = cv2.threshold(instance[0].astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
        # print(np.unique(semantic))
        _, label = cv2.threshold(y_test[i].astype(np.uint8), 0, 1, cv2.THRESH_BINARY)
        sem_iou,  sem_non_nan_iou =  f1_iou(semantic,label,2)
        # print([iou, non_nan_iou],[sem_iou, sem_non_nan_iou])
        save_score([iou, non_nan_iou],os.path.join(save_path,name))
        save_score([sem_iou,  sem_non_nan_iou],os.path.join(save_path,f'semantic_{name}'))
        # parse_and_write_metrics(os.path.join(save_path,name), str(metrics))

        cv2.imwrite(os.path.join(save_pred_path,f'{i + 1:05d}.png'), instance[0])
        cv2.imwrite(os.path.join(save_pred_nor_path,f'{i + 1:05d}.png'),  (instance[0] * (255/(np.max(instance[0])+1))).astype(np.uint8))
    # acc_csv(save_path)

def predict(model_path,test_path, save_path):
    # 保存済みのモデルを読み込む
    name = 'metrics4.csv'
    # model_path = "stardist/mymodel/2D_versatile_fluo"  # モデルの保存ディレクトリへのパス
    model = StarDist2D(model_path, name='stardist_model', basedir=save_path)
    test_img, test_label = load_images_and_labels(test_path)
    save_pred_path = os.path.join(save_path,'pred_image')
    save_pred_nor_path = os.path.join(save_path,'pred_normalize_image')
    os.makedirs(save_pred_path,exist_ok=True)
    os.makedirs(save_pred_nor_path,exist_ok=True)

    for i, test_img in enumerate(test_img):
        instance = model.predict_instances(
            img = test_img, 
        )
        # metrics =  matching(test_label, instance[0])
        # parse_and_write_metrics(save_path+'/'+name, str(metrics))
        cv2.imwrite(os.path.join(save_pred_path,f'{i + 1:05d}.png'), instance[0])
        if np.max(instance[0]) == 0:
            cv2.imwrite(os.path.join(save_pred_nor_path, f'{i + 1:05d}.png'),  instance[0])
        else:
            cv2.imwrite(os.path.join(save_pred_nor_path, f'{i + 1:05d}.png'),  (instance[0] * (255//(np.max(instance[0])))))
    # acc_csv(save_path,name)

    # モデルの保存
    # model.save("tif_image_segmentation_model.h5")

def cal_score(image, label, savepath):
    #semantic iou, recall, precision

    #instance iou, recall, precision
    return image

def main():
    path =  '/home/nakajima/work/Ecoli/code/stardist/scaledata/256_16'
    savepath1 = '/home/nakajima/work/Ecoli/code/stardist/mymodel/pretrain_2D_versatile_fluo_256'
    savepath2 = '/home/nakajima/work/Ecoli/code/stardist/mymodel/2D_versatile_fluo_256_traincolor'
    folder_list = [["train_manual", "val_manual", "test_manual2"],["train_manual_class", "val_manual_class", "test_manual2_class"]]
    training_stardist(path,savepath1, folder_list[0])
    # for i in range(len(folder_list)):
        # if i == 0:
            # training_stardist(path, savepath1, folder_list[i])
        # else:
        #     training_stardist(path, savepath2, folder_list[i])
        # input_dir = '/home/nakajima/work/Ecoli/data/nd049_label_image/'
    # image_path = input_dir + '8bit_nd049_S1_C1_T0.tif' # 8bit Grayscale画像
    # label_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_class.tif'
    # save_path = "/home/nakajima/work/Ecoli/code/stardist/mymodel/2D_versatile_fluo/nd049_S1_C1_T0"
    # predict(image_path, label_path, save_path)

if __name__ == "__main__":
    main() 