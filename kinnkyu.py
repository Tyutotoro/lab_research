import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import os
from PIL import Image
from scipy.stats import gaussian_kde


def process_images(folder_path, data_list):
    # フォルダ内のファイルを取得
    file_list = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', 'tif', 'tiff'))]
    
    # 255ピクセルが含まれる画像の数
    images_with_255 = 0
    pixel_counts = []

    for file_name in file_list:
        # 画像を読み込む
        image_path = os.path.join(folder_path, file_name)
        image = tifffile.imread(image_path) # グレースケールに変換
        # 255のピクセルをカウント
        count_255 = np.sum(image == 255)
        count_255 = count_255/ (image.shape[0]* image.shape[1])
        pixel_counts.append(count_255)
        
        # 255のピクセルが1つ以上ある場合
        if count_255 > 0:
            images_with_255 += 1
    pixel_counts_np = np.array(pixel_counts)
    pixel_mean= np.mean(pixel_counts)
    pixel_var =  np.var(pixel_counts_np)
    pixel_sqrt = np.sqrt(pixel_var)
    print(np.median(pixel_counts_np))
    plt.title('height')
    plt.grid() 
    plt.boxplot(pixel_counts_np)
    plt.show()

    # with open('/home/nakajima/work/Ecoli/code/Unet/data_100/hist/persent/persent_result.txt', 'a') as file:
    #     file.write(data_list + '\n')
    #     file.write('mean,'+ str(pixel_mean) + '\n')
    #     file.write('var,'+ str(pixel_var) + '\n')
    #     file.write('sqrt,'+ str(pixel_sqrt) + '\n')

    # ヒストグラムを表示
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(pixel_counts)), pixel_counts, tick_label=file_list)
    # plt.xlabel('Image Files')
    # plt.ylabel('parsent of 255 Pixels')
    # plt.title('Histogram of 255 Pixels in Images')
    # plt.xticks(rotation=45, ha='right')
    # plt.ylim(top = 0.12)
    # plt.tight_layout()
    # plt.savefig('/home/nakajima/work/Ecoli/code/Unet/data_100/hist/persent/'+ str(data_list)+ '.png')
    # 255のピクセルがある画像の枚数を表示
#     print(f"Number of images containing at least one 255 pixel: {images_with_255}")

# バイラテラルフィルターを適用する関数
def apply_bilateral_filter(input_folder, output_folder):
    """入力フォルダ内の画像にバイラテラルフィルターを適用し、出力フォルダに保存"""
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(os.listdir(input_folder))  # 名前順にソート
    for file in files:
        img_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        # 画像を読み込む
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # 読み込み失敗時はスキップ

        # バイラテラルフィルターを適用
        filtered_img = cv2.bilateralFilter(img, d=3, sigmaColor=30, sigmaSpace=10)

        # 保存
        cv2.imwrite(output_path, filtered_img)
        print(f"フィルター適用: {file} → {output_path}")


# data_list = ['256','128', '64','32','256_16','128_16','64_16']
# for i in range(len(data_list)):
#     path = '/home/nakajima/work/Ecoli/code/Unet/data_100/'
#     folder_path = path + data_list[i] + '/train/label'  # フォルダパスを適切に変更してください
#     process_images(folder_path, data_list[i])


def process(folder_path):
    # フォルダ内のファイルを取得
    save_path = '/home/nakajima/work/Ecoli/code/omnipose/data/class_label/'
    file_list = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', 'tif', 'tiff'))]
    for i, file_name in enumerate(file_list):
        # 画像を読み込む
        image_path = os.path.join(folder_path, file_name)
        image = tifffile.imread(image_path) # グレースケールに変換
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        
        tifffile.imwrite(save_path + f"{i:05d}.tif", n_labels)    

def main():    
    # folder_path = '/home/nakajima/work/Ecoli/code/omnipose/data/label/'
    # process(folder_path)
    # pre_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250508_003126_con'
    # make_graph(pre_path,pre_name, False)
    input_file = '/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/test_manual2/image'
    out_file = '/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/test_manual2_bila/image'
    apply_bilateral_filter(input_file, out_file)

if __name__ == "__main__":
    main()