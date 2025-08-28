import os
import shutil
import cv2
import numpy as np
import tifffile
# フォルダパスの設定
base_dir = "data"
image_dir = os.path.join(base_dir, "image")
label_dir = os.path.join(base_dir, "label")
print(label_dir)
image_list = os.listdir(image_dir)
print(len(image_list))
image = tifffile.imread(image_dir+'/' +image_list[100])
print(image_list[100])
print(np.unique(image))
print(image.shape)
print()
# 出力先フォルダを作成
# os.makedirs(image_dir, exist_ok=True)

# # labelフォルダ内のすべてのファイルを処理
# for filename in os.listdir(label_dir):
#     if filename.endswith(".tif"): 
#         # ファイル名を変更
#         base_name = os.path.splitext(filename)[0]
#         new_name = f"{base_name}_masks.tif"

#         # コピー先のパス
#         src_path = os.path.join(label_dir, filename)
#         dest_path = os.path.join(image_dir, new_name)

#         # ファイルをコピー
#         shutil.copy(src_path, dest_path)

# print("すべてのファイルをコピーし、名前を変更しました。")