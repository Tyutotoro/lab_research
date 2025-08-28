import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import os
import cv2
# import combin as com

from BaSCA.BaSCA3 import count_cell
# from make_graph import make_length_graph

from split_and_combin import load_image, scaleup_image, split_image_gray_adjust, reconstruct_image_gray_adjust
from DUNet_retinal_vessel_detection.pytorch_predict_dunet_imported import Predictor

def split_main(image_dir,image_name, image_num):
    tile_size = 256
    overlap_size = 16
    scaleup = 3
    use_overlap = True
    image_path = '/home/nakajima/work/Ecoli/data/nd049/' + image_name + '/'
    image_path = image_path + f'T{image_num:04d}.tiff' 
    # label_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label/image/8bit_nd049_S0_C1_T0.png'
    save_path = f'/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/scaledata/256_16/test_{image_dir}/T{image_num:04d}'
    image_save_path = os.path.join(save_path, 'image')
    label_save_path = os.path.join(save_path, 'label')
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)
    
    image = load_image(image_path)
    # label = load_image(label_path)
    label = np.zeros((1024, 7117))
    
    if scaleup > 1:
        image = scaleup_image(image,scaleup)
        label = scaleup_image(label,scaleup)

    image_tiles = split_image_gray_adjust(image,tile_size,use_overlap,overlap_size,save_path)
    label_tiles = split_image_gray_adjust(label,tile_size,use_overlap,overlap_size,save_path)
    # image_tiles = split_image_into_tiles(image, tile_size, overlap_size, use_overlap,save_path)
    # label_tiles = split_image_into_tiles(label, tile_size, overlap_size, use_overlap,save_path)
    if image_tiles is not None and label_tiles is not None:
        for j, (i_patch, l_patch) in enumerate(zip(image_tiles, label_tiles)):
                cv2.imwrite(os.path.join(image_save_path, f"{j:05d}.png"), i_patch)
                cv2.imwrite(os.path.join(label_save_path, f"{j:05d}.png"), l_patch)
    else:
        for j, i_patch in enumerate(image_tiles):
            cv2.imwrite(os.path.join(image_save_path, f"{j:05d}.png"), i_patch)
    return save_path

#seg
def seg_main(data_path, save_path):
    os.makedirs(save_path,exist_ok=True)
    config = {
    'seed': 3407,
    'gpuid': 0,
    'mode': 'gpu',
    'lr': 1e-4,
    'decay': 1e-6,
    'finetuning': False,
    'algorithm': 'deform_unet_v1',
    'inp_shape': (256, 256, 1),
    'data_path': data_path,
    'model_path': '/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/log/experiments/deform_unet_v1/Ecoli_20250712_181923',
    # 'log_path_experiment': '/home/work/deform_unet_v1',
    'save_path': save_path
    }

    predictor = Predictor(config)
    predictor.run()

# recon
def combin_main(base_path):
    metadata_path = '/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/scaledata/256_16/test_acrR_T1'
    image_path = base_path + '/pred_image'

    combined_image = reconstruct_image_gray_adjust(image_path,metadata_path)
    cv2.imwrite(os.path.join(base_path, 'large_label.png'), combined_image)

#basca
def basca_run_main(pred_path):
    #BaSCA
    pic_path = os.path.join(pred_path , 'large_label.png')
    print(pic_path)
    count_cell(pic_path, pred_path)
    #グラフの作成

def run():
    name_list = [['WT_all','8bit_nd049_S0_C1'],
                ['acrR_all','8bit_nd049_S3_C1']]
    # for i, name in enumerate(name_list):  
    name = name_list[1]
    save_base_path = f'/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/log/experiments/deform_unet_v1/Ecoli_'+ datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+f'{name[0]}' 
        # if i == 0:
        #     n = 56
        # else:
        #     n = 1
    for j in range(48,183):
        save_path = os.path.join(save_base_path, f'T{j:04d}')
        data_path = split_main(image_dir = name[0],image_name = name[1],image_num=j)
        seg_main(data_path=data_path, save_path=save_path)
        combin_main(base_path=save_path)
        basca_run_main(pred_path=save_path)


if __name__ == '__main__':
    run()