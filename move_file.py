import cv2
import numpy as np
import os
import shutil
import random
import tifffile
from getshape import Getshape
from picture import Picture
gsh = Getshape()
pic = Picture()
random.seed(0)

# 移動済みファイルを管理するセット
moved_files = set()

# フォルダ内の画像を取得してソートする
def get_sorted_label_list(label_dir):
    labels = [os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith(('.png', '.tif', '.tiff'))]
    return sorted(labels, key=lambda x: os.path.basename(x))

# 全ピクセル値が0の画像を除外する
def filter_non_empty_images(pairs):
    valid_pairs = []
    for image_file, label_file in pairs:
        try:
            label_img = tifffile.imread(label_file)
        except:
            label_img = cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)
        min_val, max_val, _, _ = cv2.minMaxLoc(label_img)
        # 全ピクセルが0でない場合のみペアを保持
        if (min_val, max_val) != (0, 0):
            valid_pairs.append((image_file, label_file))
    return valid_pairs  # 有効なペアのみで更新

def copy_files_with_optional_rename(file_list, dest_dir, rename=None):
    """
    ファイルを指定したディレクトリにコピーし、必要に応じてリネームする関数

    :param file_list: コピーするファイルのリスト [(image_path, label_path), ...]
    :param dest_dir: コピー先のディレクトリ
    :param rename: リネームするプレフィックス（デフォルトはNone）
    """
    image_dir = os.path.join(dest_dir, "image")
    label_dir = os.path.join(dest_dir, "label")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for image_file, label_file in file_list:
        # 画像をコピー
        shutil.copy(image_file, os.path.join(image_dir, os.path.basename(image_file)))

        # ラベルをコピー (rename が有効なら拡張子の前に付け足す)
        if rename:
            base_name, ext = os.path.splitext(os.path.basename(label_file))
            new_name = f"{base_name}_{rename}{ext}"
            shutil.copy(label_file, os.path.join(label_dir, new_name))
        else:
            shutil.copy(label_file, os.path.join(label_dir, os.path.basename(label_file)))



def run(train: bool, val: bool, test: bool, s_image_path: str, s_label_path: str, train_path: str, val_path: str, 
        test_path: str, ratio: any,rename: any, removed_0: bool):
    global moved_files

    #フォルダ内の画像リストを取得
    label_file_list = get_sorted_label_list(s_label_path)
    image_file_lsit = get_sorted_label_list(s_image_path)
    pairs = list(zip(image_file_lsit, label_file_list))  # ペアリスト作成
    

    if removed_0 == True:
        move_list = filter_non_empty_images(pairs)
    else: 
        move_list = pairs

    #移動するファイル数
    move_num = int(len(move_list)*ratio)

    #val の処理
    if val:
        available_pairs = [pair for pair in move_list if pair not in moved_files]
        val_selected = random.sample(available_pairs, move_num)
        copy_files_with_optional_rename(val_selected, val_path, rename)
        moved_files.update(val_selected)  # moved_files に追加

    #test の処理
    if test:
        available_pairs = [pair for pair in move_list if pair not in moved_files]
        test_selected = random.sample(available_pairs, move_num)
        copy_files_with_optional_rename(test_selected, test_path, rename)
        moved_files.update(test_selected)  # moved_files に追加

    #train の処理
    if train:
        train_remaining = [pair for pair in move_list if pair not in moved_files]
        copy_files_with_optional_rename(train_remaining, train_path, rename)

    else:
        print('error')



def main():
    #parameterの設定
    make_folder = [True, True, True] #作成するフォルダ[train, val, test]

    src_image_path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/hokan/image'
    src_label_path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/hokan/label'
    tar_path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16'
    train_path = 'train_WT'
    val_path = 'val_WT'
    test_path = 'test_WT'

    val_ratio = 0.1 #valに移動させる割合

    rename = None #labelのファイル名を変える時の変える追加する名前
    removed_0 = False #大腸菌が写っていない画像を省く



    run(train = make_folder[0],
        val = make_folder[1],
        test = make_folder[2],
        s_image_path = src_image_path,
        s_label_path = src_label_path,
        train_path = os.path.join(tar_path, train_path),
        val_path = os.path.join(tar_path, val_path),
        test_path = os.path.join(tar_path, test_path),
        ratio = val_ratio,
        rename = rename,
        removed_0 = removed_0
        )


if __name__ == "__main__":
    main()


# #リストに基づき画像をコピー
# def copy_images(image_list, src_dir, dest_dir, folder_name):
#     dest_folder = os.path.join(dest_dir, folder_name)
#     os.makedirs(dest_folder, exist_ok=True)
#     for src_path in image_list:
#         base_name = os.path.basename(src_path)
#         shutil.copy(os.path.join(src_dir, base_name), os.path.join(dest_folder, base_name))


# # コピー用の関数を定義
# def copy_and_rename_files(src_dir, dst_dir, start_index=0, rename_prefix=None, extension=None):
#     """
#     ファイルを指定したディレクトリにコピーし、必要に応じてリネームする関数

#     :param src_dir: コピー元ディレクトリ
#     :param dst_dir: コピー先ディレクトリ
#     :param start_index: リネーム時の開始インデックス（デフォルトは0）
#     :param rename_prefix: リネーム時のファイル名プレフィックス（デフォルトはNone）
#     :param extension: ファイル名変更時の拡張子（デフォルトはNone）
#     """
#     os.makedirs(dst_dir, exist_ok=True)
#     files = sorted(os.listdir(src_dir))
#     print(len(files))

#     for i, filename in enumerate(files):
#         src = os.path.join(src_dir, filename)
#         if rename_prefix:
#             new_name = f"{start_index + i:05d}"
#             if extension:
#                 new_name += f".{extension}"
#             else:
#                 new_name += os.path.splitext(filename)[1]  # 元の拡張子を保持
#             dst = os.path.join(dst_dir, new_name)
#         else:
#             dst = os.path.join(dst_dir, filename)
#         shutil.copy(src, dst)


# def move_train_to_val(src_image_dir, src_label_dir, tgt_image_dir, tgt_label_dir, num_to_move: int, moved_files: set):
#     # 画像名のペアを取得
#     image_files = sorted(os.listdir(src_image_dir))  # 生画像
#     label_files = sorted(os.listdir(src_label_dir))  # ラベル画像
#     pairs = list(zip(image_files, label_files))  # ペアリスト作成

#     # 移動対象のペアをフィルタリング（未移動のものだけを選択）
#     available_pairs = [pair for pair in pairs if pair[0] not in moved_files]

#     # 移動するペアの数を調整（未移動ペアが指定数より少ない場合に対応）
#     num_to_move = min(num_to_move, len(available_pairs))

#     # ランダム選択
#     pairs_to_move = random.sample(available_pairs, num_to_move)

#     for image_file, label_file in pairs_to_move:
#         shutil.move(os.path.join(src_image_dir, image_file), os.path.join(tgt_image_dir, image_file))
#         shutil.move(os.path.join(src_label_dir, label_file), os.path.join(tgt_label_dir, label_file))
#         moved_files.add(image_file)  # 移動済みとして記録


# # フォルダのパスを設定
# train_image_dir = "/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/train/image"
# val_image_dir = "/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/val/image"
# train_label_dir = "/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/train/label"
# val_label_dir = "/home/nakajima/work/Ecoli/code/Unet/scaledata/256_16/val/label"
# scale_image_dir = "/home/nakajima/work/Ecoli/code/omnipose/data/scaleimageraw256"
# scale_label_dir = "/home/nakajima/work/Ecoli/code/omnipose/data/scalelabelraw256"
# image_dir = "/home/nakajima/work/Ecoli/code/omnipose/data/scaleimage256"

# copy_and_rename_files(train_label_dir, scale_label_dir)
# train_label_count = len(os.listdir(train_label_dir))
# copy_and_rename_files(val_label_dir, scale_label_dir, start_index=train_label_count, rename_prefix="label", extension="png")

# copy_and_rename_files(train_image_dir, scale_image_dir)
# train_image_count = len(os.listdir(train_image_dir))
# copy_and_rename_files(val_image_dir, scale_image_dir, start_index=train_image_count, rename_prefix="image", extension="png")


# copy_and_rename_files(scale_image_dir,image_dir)
# copy_labels_to_images_with_masks(scale_label_dir, image_dir)