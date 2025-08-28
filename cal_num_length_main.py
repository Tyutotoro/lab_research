import os
import cv2
import re
import combin as com

from BaSCA.BaSCA3 import count_lengh_cal
from make_graph import make_length_graph

def comb_main(base_path,img_size):
    image_folder = os.path.join(base_path,"pred_image") 
    # image_folder = base_path 
    save_name = "large_label.png"
    
    labels = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".png", ".tif", "tiff"))]
    image_paths = sorted(labels, key=lambda s: int(re.search(r'(\d+)\.', s).groups()[0]))

    if img_size == 3:    
        label_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_label_large.tif"
        raw_image_path = "/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/test_manual2/large_image.png"
        combine_image_size = [3072, 21351]  # 完成後の画像サイズ (幅, 高さ)(ノーマル[1024,7117], 3x3 [3072, 21351])
        overlap = 16               # 上下左右の重ねるピクセル数
        left_size = 226            # 左側の埋めるピクセル数　( 3x3 226)
        right_size = 229           # 右側の埋めるピクセル数　( 3x3 229)(3x3 の128x128のみ165)

    elif img_size == 1:
        label_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_label.tif'
        raw_image_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S1_C1_T0.tif'
        combine_image_size = [1024, 7117]  # 完成後の画像サイズ (幅, 高さ)(ノーマル[1024,7117], 3x3 [3072, 21351])
        overlap = 16               # 上下左右の重ねるピクセル数
        left_size = 94          # 左側の埋めるピクセル数　(ノーマル 94) 
        right_size = 63           # 右側の埋めるピクセル数　(ノーマル 95)(ノーマル の128x128のみ63, ノーマル　の256x256のみ47)
    else:
        print("error")
    combined_image = com.combine_image(image_paths= image_paths, combine_image_size=combine_image_size,overlap=overlap,
                        left_size=left_size, right_size=right_size, output_path=os.path.join(base_path,save_name))
    true_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

    ##overlapするとき
    com.make_overlap_image(true_label, combined_image,combine_image_size,base_path)
    com.calculation_score(true_label=true_label, combined_image=combined_image,base_path=base_path)

def main(pred_name, img_size):
    pred_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/"
    # pred_path = '/home/nakajima/work/Ecoli/code/Unet/result/'
    base_path = pred_path + pred_name
    # base_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/data/64_16/test_manual2/label'
    #画像の統合
    # comb_main(base_path,img_size)
    #BaSCA
    pic_path = os.path.join(base_path , 'large_label.png')
    
    # base_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/'
    # pic_path = base_path + pred_name
    print(pic_path)
    count_lengh_cal(pic_path, base_path)
    #グラフの作成
    list_name = '/length_list_gmmmml3.csv'
    if img_size == 1:
        make_length_graph(base_path, list_name, False)
    elif img_size == 3:
        # make_length_graph(base_path, list_name, True)
        None
    else:
        print('image size error')

if __name__ == '__main__':
    path_list = [
        #normal
        # '20250402_173310_64_16_con',
        # '20250613_140836_128_16_con',
        # '20250613_132000_256_16_con',
        
        #large
        # '20250404_080149_64_16_con',
        # '20250404_011247_128_16_con',
        # '20250403_193137_256_16_con',

        #normal
        # 'Ecoli_20250610_201918_64normal_nopre_con',
        # 'Ecoli_20250611_222449_128normal_nopre_con',
        # 'Ecoli_20250612_123339_256normal_nopre_con',

        #large
        # 'Ecoli_20250608_181159_64_nopre_con',
        # 'Ecoli_20250609_172927_128_nopre_con',
        # 'Ecoli_20250508_003126_256_nopre_con',
        # 'Ecoli_20250514_173835_256_CLAHE_con',
        # 'Ecoli_20250528_031059_256_high_con',
        # 'Ecoli_20250604_125126_256_bila_test_con/test',

        # 'Ecoli_20250713_164357/test',
        # 'Ecoli_20250713_164701/test',
        # '8bit_nd049_S1_C1_T0_label.tif',
        # '8bit_nd049_S1_C1_T0_label_large.tif'
        # 'Ecoli_20250528_031059_256_high_con',
        # 'Ecoli_20250715_171539_arcA_T30/test',
        # 'Ecoli_20250715_160750_arcB_T30/test',
        # 'Ecoli_20250715_160619_arcR_T30/test',
        # 'Ecoli_20250715_161016_WT_T30/test',
        # 'Ecoli_20250722_144023_WT_T0/test'
        'Ecoli_20250715_163717_manual2/test'
                ]
    for i in range(len(path_list)):
        main(path_list[i],3)