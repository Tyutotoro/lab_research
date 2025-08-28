import cv2
import numpy as np
import os
import shutil
import random
from getshape import Getshape

gsh = Getshape()
random.seed(0)

def enlarge_image(image):
    """1ピクセルを3x3に拡大する関数"""
    height, width = image.shape
    new_height, new_width = height * 3, width * 3
    resized_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            value = image[y, x]
            resized_image[y*3:y*3+3, x*3:x*3+3] = value  # 3×3のブロックに展開

    return resized_image

##画像のトリミング&分割
def trim_and_split_image(input_image, trim_left=226, trim_right=229, target_size = None, overlap =False, oversize= None):
    width = input_image.shape[1] - trim_right
    height = input_image.shape[0]
    trimmed_image = input_image[0 : height, trim_left : width]
    # トリミングした画像を均等に分割
    if overlap:
        slice_image = over_split(trimmed_image, target_size, oversize)
        print('over_split')
    else:
        slice_image = gsh.split(trimmed_image,sizeh=target_size, sizev=target_size)

    slice_image_np  = np.array(slice_image)
    return slice_image

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
                if tile_y_end - y < tile_size:
                    print('tile over y')
                    print(tile_y_end)
                    print(tile_y_end - y)
                if tile_x_end - x < tile_size:
                    print('tile over x')
                    print(tile_x_end)
                    print(tile_x_end - x)
                y_overlap = tile_size - (tile_y_end - y)
                x_overlap = tile_size - (tile_x_end - x)
                tile = image[max(0, y - y_overlap):tile_y_end, max(0, x - x_overlap):tile_x_end]

            tiles.append(tile)
    return tiles

def trim_and_slice(image,target_size = None, overlap =False, oversize= None):
    height, width = image.shape[0],image.shape[1] 
    # トリミングした画像を均等に分割
    if overlap:
        print(height,width)
        slice_image = []
        x_over = 0
        y_over = 0
        y_steps = range(0, height - oversize, target_size - oversize)
        x_steps = range(0, width - oversize, target_size - oversize)
        
        # 画像を分割する
        for y in y_steps:
            for x in x_steps:
                # タイルのサイズが画像の端を超える場合に、端の部分を重複させる
                tile_y_end = min(y + target_size, height)
                tile_x_end = min(x + target_size, width)
                # print(f'y:y,x:x {y,tile_y_end, x,tile_x_end}')
                tile = image[y:tile_y_end, x:tile_x_end]

                # 縦または横が足りない場合、重複部分を増やしてタイルサイズを合わせる
                if tile_y_end - y < target_size or tile_x_end - x < target_size:
                    if tile_y_end - y < target_size:
                        print('tile over y')
                        print(tile_y_end)
                        print(tile_y_end - y)
                        y_over = tile_y_end - y
                    if tile_x_end - x < target_size:
                        print('tile over x')
                        print(tile_x_end)
                        print(tile_x_end - x)
                        x_over = tile_x_end - x
                    y_overlap = target_size - (tile_y_end - y)
                    x_overlap = target_size - (tile_x_end - x)
                    print(x_overlap,y_overlap)
                    tile = image[max(0, y - y_overlap):tile_y_end, max(0, x - x_overlap):tile_x_end]
                slice_image.append(tile)
        print('over_split')
    else:
        slice_image = gsh.split(image,sizeh=target_size, sizev=target_size)
    slice_image_np  = np.array(slice_image)
    return slice_image, x_over, y_over

# img = cv2.imread('/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S0_C1_T0.tiff', cv2.IMREAD_GRAYSCALE)
# re_image = enlarge_image(img)
# print(img.shape)
# print(re_image.shape)

# sp_img = trim_and_split_image(input_image=re_image, target_size=256, overlap=True, oversize=16)
# print(sp_img)


# def move_train_to_val(src_image_dir, src_label_dir, tgt_image_dir, tgt_label_dir, num_to_move:int):
#     # 画像名のペアを取得
#     image_files = sorted(os.listdir(src_image_dir))  # グレー画像
#     label_files = sorted(os.listdir(src_label_dir))  # 2値画像
#     pairs = list(zip(image_files, label_files))  # ペアリスト作成

#     # 移動するペアの数（最低1組）
#     pairs_to_move = random.sample(pairs, num_to_move)  # ランダム選択

#     for image_file, label_file in pairs_to_move:
#         shutil.move(os.path.join(src_image_dir, image_file), os.path.join(tgt_image_dir, image_file))
#         shutil.move(os.path.join(src_label_dir, label_file), os.path.join(tgt_label_dir, label_file))

def move_train_to_val(src_image_dir, src_label_dir, tgt_image_dir, tgt_label_dir, num_to_move: int, moved_files: set):
    # 画像名のペアを取得
    image_files = sorted(os.listdir(src_image_dir))  # グレー画像
    label_files = sorted(os.listdir(src_label_dir))  # 2値画像
    pairs = list(zip(image_files, label_files))  # ペアリスト作成

    # 移動対象のペアをフィルタリング（未移動のものだけを選択）
    available_pairs = [pair for pair in pairs if pair[0] not in moved_files]

    # 移動するペアの数を調整（未移動ペアが指定数より少ない場合に対応）
    num_to_move = min(num_to_move, len(available_pairs))

    # ランダム選択
    pairs_to_move = random.sample(available_pairs, num_to_move)

    for image_file, label_file in pairs_to_move:
        shutil.move(os.path.join(src_image_dir, image_file), os.path.join(tgt_image_dir, image_file))
        shutil.move(os.path.join(src_label_dir, label_file), os.path.join(tgt_label_dir, label_file))
        moved_files.add(image_file)  # 移動済みとして記録


def rename_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"フォルダが見つかりません: {folder_path}")
        return

    # ファイルリストを取得してソート
    files = sorted(os.listdir(folder_path))

    for idx, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]  # 拡張子を取得
        new_name = f"{idx:05d}{ext}"  # 5桁ゼロ埋め
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        # print(f"リネーム: {filename} → {new_name}")

def main():
    
    # 画像の処理と保存
    patch_size = 256 # 分割するパッチサイズ
    moved_files = set()
    train_name = 'train_manual'
    val_name = 'val_manual'
    test_name = 'test_manual2'
    save_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/'
    # trim_left =226 
    # trim_right =229 
    trim_left =94  
    trim_right =47
    size = str(patch_size)


    # 入力画像のペア（グレー画像と2値画像）
    image_pairs = [
        ('/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S0_C1_T0.tiff', '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label/image/8bit_nd049_S0_C1_T0.png'),
        ('/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S0_C1_T1.tiff', '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label/image/8bit_nd049_S0_C1_T1.png'),
        ('/home/nakajima/work/Ecoli/data/nd049_label_image/8bit_nd049_S1_C1_T0.tif', '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/8bit_nd049_S1_C1_T0_label.tif')
        # ('/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label_mass/8bit_nd049_S1_C1_T33_split.tif','/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label_mass/8bit_nd049_S1_C1_T33_split_label.tif')
    ]


    # 保存先フォルダ
    output_dirs = {
        "train_image": save_path + size +"_16/"+ train_name + "/image",
        "train_label": save_path + size +"_16/"+train_name+"/label",
        "val_image": save_path + size +"_16/"+val_name+"/image",
        "val_label": save_path + size +"_16/"+val_name+"/label",
        "test_image": save_path + size +"_16/"+test_name+"/image",
        "test_label": save_path + size +"_16/"+test_name+"/label"
    }

    # 出力フォルダを作成
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    for i, (gray_path, binary_path) in enumerate(image_pairs):
        # 画像をグレースケールで読み込む
        print(gray_path, binary_path)
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

        # 画像を拡大
        # gray_resized = enlarge_image(gray_img)
        # binary_resized = enlarge_image(binary_img)
        gray_resized = gray_img
        binary_resized = binary_img

        # 画像を分割
        gray_patches = trim_and_split_image(gray_resized, trim_left=trim_left, trim_right=trim_right, target_size = patch_size, overlap =True, oversize= 16)
        binary_patches = trim_and_split_image(binary_resized, trim_left=trim_left, trim_right=trim_right, target_size = patch_size, overlap =True, oversize= 16)

        # 保存先の選択（最初の2組はtrain、最後の1組はtest）
        if i < 2:
            img_dir = output_dirs["train_image"]
            label_dir = output_dirs["train_label"]
        else:
            img_dir = output_dirs["test_image"]
            label_dir = output_dirs["test_label"]

        # 各パッチを保存
        for j, (g_patch, b_patch) in enumerate(zip(gray_patches, binary_patches)):
            cv2.imwrite(os.path.join(img_dir, f"pair{i+1}_patch{j:05d}_gray.png"), g_patch)
            cv2.imwrite(os.path.join(label_dir, f"pair{i+1}_patch{j:05d}_binary.png"), b_patch)
    

    # 各フォルダをリネーム
    rename_files_in_folder(output_dirs["train_image"])
    rename_files_in_folder(output_dirs["train_label"])
    rename_files_in_folder(output_dirs["test_image"])
    rename_files_in_folder(output_dirs["test_label"])

    num = int(len(os.listdir(output_dirs["train_label"]))*0.1)
    move_train_to_val(src_image_dir= output_dirs["train_image"], 
                    src_label_dir=output_dirs["train_label"], 
                    tgt_image_dir=output_dirs["val_image"],
                    tgt_label_dir= output_dirs["val_label"],
                    num_to_move=num,
                    moved_files=moved_files)


# # バイラテラルフィルターを適用する関数
# def apply_bilateral_filter(input_folder, output_folder):
#     """入力フォルダ内の画像にバイラテラルフィルターを適用し、出力フォルダに保存"""
#     os.makedirs(output_folder, exist_ok=True)

#     files = sorted(os.listdir(input_folder))  # 名前順にソート
#     for file in files:
#         img_path = os.path.join(input_folder, file)
#         output_path = os.path.join(output_folder, file)

#         # 画像を読み込む
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue  # 読み込み失敗時はスキップ

#         # バイラテラルフィルターを適用
#         filtered_img = cv2.bilateralFilter(img, d=3, sigmaColor=30, sigmaSpace=10)

#         # 保存
#         cv2.imwrite(output_path, filtered_img)
#         print(f"フィルター適用: {file} → {output_path}")

# # ラベルをコピーする関数
# def copy_labels(input_folder, output_folder):
#     """ラベルフォルダの内容をそのままコピー"""
#     os.makedirs(output_folder, exist_ok=True)

#     files = sorted(os.listdir(input_folder))  # 名前順にソート
#     for file in files:
#         src_path = os.path.join(input_folder, file)
#         dst_path = os.path.join(output_folder, file)
#         shutil.copy2(src_path, dst_path)  # メタデータを維持してコピー
#         print(f"ラベルコピー: {file} → {dst_path}")

# # フォルダパスの定義
# path = "Unet/scaldata/"+"128_16"
# train_image_folder = path+ "/train/image"
# train_label_folder = path+"/train/label"
# bila_image_folder = path+"/train_bila/image"
# bila_label_folder = path+"/train_bila/label"

# # フィルター適用とラベルコピー
# apply_bilateral_filter(train_image_folder, bila_image_folder)
# copy_labels(train_label_folder, bila_label_folder)

if __name__ == '__main__':
    main()