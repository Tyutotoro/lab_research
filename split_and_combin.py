import cv2
import numpy as np
import os
import tifffile
import json
import matplotlib.pyplot as plt

#画像読み込み
def load_image(path):
    try:
        input_image = tifffile.imread(path)
    except:
        input_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return input_image

def scaleup_image(image,scaleup):
    if image is None:
        return None
    height, width = image.shape
    new_height, new_width = height * scaleup, width * scaleup
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            value = image[y, x]
            resized_image[y*scaleup:y*scaleup+scaleup, x*scaleup:x*scaleup+scaleup] = value  # 3×3のブロックに展開
    return resized_image





def split_image_gray_adjust(img, tile=256, overlap=True, ov=32, save_dir='tiles'):
    if img is None:
        print('image is None')
        return None
    # os.makedirs(save_dir, exist_ok=True)
    h, w = img.shape

    step = tile - ov if overlap else tile
    metadata = {
        'original_size': (w, h),
        'tile_size': tile,
        'overlap': overlap,
        'overlap_size': ov,
        'adjustments': []  # 拡張したオーバーラップ情報
    }

    tile_list = []
    y_positions = []
    x_positions = []

    # Y位置の計算
    y = 0
    while y + tile <= h:
        y_positions.append(y)
        y += step
    if y < h:
        # 最後のタイルを上側オーバーラップで調整
        adj = tile - (h - y)
        y = h - tile
        y_positions.append(y)
        metadata['adjustments'].append({'index': 'y_last', 'adjust_top': adj})

    # X位置の計算
    x = 0
    while x + tile <= w:
        x_positions.append(x)
        x += step
    if x < w:
        # 最後のタイルを左側オーバーラップで調整
        adj = tile - (w - x)
        x = w - tile
        x_positions.append(x)
        metadata['adjustments'].append({'index': 'x_last', 'adjust_left': adj})

    for y in y_positions:
        for x in x_positions:
            crop = img[y:y + tile, x:x + tile]
            tile_list.append(crop)
            # cv2.imwrite(os.path.join(save_dir, f'{index}.png'), crop)
            # index += 1

    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    return tile_list


def reconstruct_image_gray_adjust(tile_dir, meta_path):
    with open(os.path.join(meta_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    w, h = meta['original_size']
    tile = meta['tile_size']
    ov = meta['overlap_size'] if meta['overlap'] else 0
    step = tile - ov if meta['overlap'] else tile

    reconstructed = np.zeros((h, w), dtype=np.uint16)
    index = 0
    y_positions = []
    x_positions = []

    # Y再構築位置
    y = 0
    while y + tile <= h:
        y_positions.append(y)
        y += step
    if y < h:
        y = h - tile
        y_positions.append(y)

    # X再構築位置
    x = 0
    while x + tile <= w:
        x_positions.append(x)
        x += step
    if x < w:
        x = w - tile
        x_positions.append(x)

    for y in y_positions:
        for x in x_positions:
            tile_path = os.path.join(tile_dir, f'{index:05d}.png')
            if not os.path.exists(tile_path):
                index += 1
                continue
            tile_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            reconstructed[y:y + tile, x:x + tile] += tile_img
            index += 1

    result = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return result


def main(split):
    if split:
        tile_size = 256
        overlap_size = 16
        scaleup = 3
        use_overlap = True
        image_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S0_C1_T0.tiff'
        label_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label/image/8bit_nd049_S0_C1_T0.png'
        save_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/scaledata/256_16/test_WT_T0'
        image_save_path = os.path.join(save_path, 'image')
        label_save_path = os.path.join(save_path, 'label')
        
        os.makedirs(image_save_path, exist_ok=True)
        os.makedirs(label_save_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        
        image = load_image(image_path)
        label = load_image(label_path)
        # label = np.zeros((1024, 7117))
        
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

    else:
        base_path  = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250722_144023'
        metadata_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/scaledata/256_16/test_WT_T0'
        image_path = base_path + '/test/pred_image'
        save_path = base_path + '/test'

        # image_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/scaledata/256_16/test_arcR/image'
        # save_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/scaledata/256_16/test_arcR'

        # metadata_file_path = metadata_path + '/image_metadata.txt'
        # combined_image =combin_image_from_metadata(image_path, metadata_file_path)
        combined_image = reconstruct_image_gray_adjust(image_path,metadata_path)
        cv2.imwrite(os.path.join(save_path, 'large_label.png'), combined_image)

if __name__ == "__main__":
    # split = True
    split = False
    main(split)