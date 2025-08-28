import cv2
import numpy as np
import os
import re
from scalup import enlarge_image
from cal_IoU import f1_IoU
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from getshape import Getshape
gsh = Getshape()

def combine_image(image_paths, combine_image_size, overlap, left_size, right_size, output_path):
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    if not images or images[0] is None:
        print("画像が読み込めません。")
        return
    tile_size = np.array(images)[0].shape
    combined_image = np.zeros((combine_image_size[0], combine_image_size[1]), dtype=np.uint8) 

    current_x = left_size  
    current_y = 0          
    i = 0
    for img in images:
        i = i+1
        # 右端を超えた場合は次の行に移動
        if current_x + tile_size[0] > combine_image_size[1] - right_size:  
            print(f'{current_y},{current_x} x: {combine_image_size[1]-right_size}, {i}')
            current_x = left_size
            current_y += tile_size[1] - overlap


        # 下端を超えた場合の調整
        if current_y + tile_size[1] > combine_image_size[0]: 
            excess_height = (current_y + tile_size[1]) - combine_image_size[0]
            adjusted_overlap = overlap + excess_height
            current_y -= (adjusted_overlap - overlap)  

        x_end = current_x + tile_size[0]
        y_end = current_y + tile_size[1]



        if y_end <= combine_image_size[0]:  
            combined_image[current_y:y_end, current_x:x_end] = img[:tile_size[1], :tile_size[0]]

        current_x += tile_size[0] - overlap

    print(f'height{current_y}, width {current_x}')
    print(f'i : {i}')
    valid_height = current_y + tile_size[1] if current_y + tile_size[1] <= combine_image_size[0] else combine_image_size[0]
    trimmed_combine_image = combined_image[:valid_height, :]
    cv2.imwrite(output_path, trimmed_combine_image)

    return combined_image

def combine_image2(image_paths, combine_image_size, overlap, left_size, right_size, output_path):
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    if not images or images[0] is None:
        print("画像が読み込めません。")
        return
    tile_size = np.array(images)[0].shape
    combined_image = np.zeros((combine_image_size[0], combine_image_size[1]), dtype=np.uint8) 

    current_x = left_size  
    current_y = 0          
    i = 0
    for img in images:
        i = i+1
        # 右端を超えた場合は次の行に移動
        if current_x + tile_size[0] > combine_image_size[1] - right_size:  
            print(f'{current_y},{current_x} x: {combine_image_size[1]-right_size}, {i}')
            current_x = left_size
            current_y += tile_size[1] - overlap


        # 下端を超えた場合の調整
        if current_y + tile_size[1] > combine_image_size[0]: 
            excess_height = (current_y + tile_size[1]) - combine_image_size[0]
            adjusted_overlap = overlap + excess_height
            current_y -= (adjusted_overlap - overlap)  

        x_end = current_x + tile_size[0]
        y_end = current_y + tile_size[1]



        if y_end <= combine_image_size[0]:  
            combined_image[current_y:y_end, current_x:x_end] = img[:tile_size[1], :tile_size[0]]

        current_x += tile_size[0] - overlap

    print(f'height{current_y}, width {current_x}')
    print(f'i : {i}')
    valid_height = current_y + tile_size[1] if current_y + tile_size[1] <= combine_image_size[0] else combine_image_size[0]
    trimmed_combine_image = combined_image[:valid_height, :]
    cv2.imwrite(output_path, trimmed_combine_image)

    return combined_image



def calculate_pixel_match_ratio(image1, image2):
    if image1 is None or image2 is None:
        print("画像が正しく読み込まれていません。")
        return None

    if image1.shape != image2.shape:
        print("画像のサイズが一致していません。")
        return None

    match_mask = image1 == image2

    match_count = np.sum(match_mask)
    total_pixels = image1.size

    match_ratio = match_count / total_pixels
    print(match_ratio)
    return match_ratio

def color_count(image):
    from collections import Counter
    pixels = image.reshape(-1, 3)  # (height*width, 3)
    pixel_tuples = [tuple(pixel) for pixel in pixels]

    # 各カラーの出現回数をカウント
    color_counts = Counter(pixel_tuples)

    # カラーとその個数を出力
    print(f"カラーの総数: {len(color_counts)}")
    for color, count in color_counts.items():
        print(f"カラー (RGB): {list(color)} - 個数: {count}")

def make_overlap_image(truelabel, cover_image,combine_image_size,base_path):
    #overlapするときの関数
    #白色が正解ラベル
    #青色が予測結果
    true_label = np.where(truelabel == 0, 200, truelabel)
    background = np.zeros((combine_image_size[0], combine_image_size[1]), dtype=np.uint8)
    cover_image_color = np.array([background,background,cover_image]).transpose(1,2,0)
    true_label_color = np.array([true_label, true_label,true_label]).transpose(1,2,0)
    overlap_image = gsh.overlay_img(raw_img=true_label_color,proce_img=cover_image_color, num= [0.8,0.2,0] )
    overlap_image_rgb = overlap_image[:, :, [2, 1, 0]]
    overlap_image_rgb = np.where((overlap_image_rgb == [211,160,160]).all(axis=-1)[..., None],[160,255,255] ,overlap_image_rgb)#外れてる箇所
    overlap_image_rgb= np.where((overlap_image_rgb ==  [255,204,204]).all(axis=-1)[..., None], [0,0,255], overlap_image_rgb)#当たっている箇所
    overlap_image_rgb= np.where(overlap_image_rgb == 204, 255, overlap_image_rgb)#正解ラベルのみの箇所
    overlap_image_bgr = overlap_image_rgb[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(base_path,'overlap_image.png'),overlap_image_bgr)

def calculation_score(true_label,combined_image,base_path):
    combined_image = (combined_image/255).astype('uint8')
    true_label = (true_label/255).astype('uint8') 
    iou, f1 = f1_IoU(pred=combined_image, true=true_label)
    print(f'iou: {iou}')
    write_score(iou,f1, base_path)


def write_score(iou, f1, save_path):
    with open(save_path + '/' + 'score.txt', 'w') as f:
            f.write(f'class1: {iou[0]},'
                    f'class2: {iou[1]},'
                    f'mean_score: {iou[2]},'
                    f'recall: {f1[0]},'
                    f'precision: {f1[1]},'
                    f'f1-measure: {f1[2]},'
                    '\n')
            



def main():
    # base_path = "/home/nakajima/work/Ecoli/code/Unet/result/20250224_012822_256_16"
    base_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/"
    base_name = "Ecoli_20250715_163717_manual2/test"
    base_path = base_path + base_name
    image_folder = os.path.join(base_path,"pred_image")  
    save_name = "large_label.png"
    label_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_label_large.tif"
    raw_image_path = "/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/test_manual2/large_image.png"


    labels = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith((".png", ".tif", "tiff"))]
    image_paths = sorted(labels, key=lambda s: int(re.search(r'(\d+)\.', s).groups()[0]))

    combine_image_size = [3072, 21351]  # 完成後の画像サイズ (幅, 高さ)(ノーマル[1024,7117], 3x3 [3072, 21351])
    overlap = 16               # 上下左右の重ねるピクセル数
    left_size = 226            # 左側の埋めるピクセル数　(ノーマル 94, 3x3 226)
    right_size = 229           # 右側の埋めるピクセル数　(ノーマル 95, 3x3 229)(3x3 の128のみ165)
    combined_image = combine_image(image_paths= image_paths, combine_image_size=combine_image_size,overlap=overlap,
                                left_size=left_size, right_size=right_size, output_path=os.path.join(base_path,save_name))
    # combined_image = create_tiled_image_with_gif(image_paths=image_paths, canvas_size=combine_image_size, overlap=overlap,
    #                                              left_size=left_size, right_size=right_size, output_image_path=os.path.join(base_path,save_name),
    #                                              gif_path=os.path.join(base_path,"output_tiling_process.gif" ))
    true_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

    ##overlapするとき
    make_overlap_image(true_label, combined_image,combine_image_size,base_path)
    calculation_score(true_label=true_label, combined_image=combined_image,base_path=base_path)
    # calculate_pixel_match_ratio(true_label,combined_image,base_path)

if __name__ == "__main__":
    main()