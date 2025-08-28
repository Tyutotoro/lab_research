import cv2
from cv2 import dnn_superres
import os
from tqdm import tqdm
import re
from getshape import Getshape
gsh = Getshape()


def create_edsr_superres_folder(base_folder, name, model_path, scale, source_folder):
    # [name]_EDSRフォルダとサブフォルダの作成
    edsr_folder = f"{name}"
    image_output_folder = os.path.join(base_folder,edsr_folder, "image")
    label_output_folder = os.path.join(base_folder,edsr_folder, "label")

    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(label_output_folder, exist_ok=True)

    print(f"フォルダ {image_output_folder} が作成されました。")

    # dnn_superresオブジェクトを作成
    sr = dnn_superres.DnnSuperResImpl_create()

    # モデルの読み込みと設定
    sr.readModel(model_path)
    sr.setModel("edsr", scale)

    # 元の image と label フォルダのパス
    image_source_folder = os.path.join(source_folder, "image")
    label_source_folder = os.path.join(source_folder, "label")

    # 超解像を実行して保存
    for folder, output_folder in [(image_source_folder, image_output_folder), 
                                (label_source_folder, label_output_folder)]:
        if not os.path.exists(folder):
            print(f"フォルダ {folder} が存在しません。処理をスキップします。")
            continue
        
        for filename in tqdm(os.listdir(folder)):
            if filename.endswith((".png", ".tiff", ".tif")):  # 対応する画像フォーマットを確認
                input_path = os.path.join(folder, filename)
                output_path = os.path.join(output_folder, filename)

                # 画像を読み込む
                image = cv2.imread(input_path)
                if image is None:
                    print(f"画像 {filename} を読み込めませんでした。")
                    continue

                # 超解像の実行
                result = sr.upsample(image)

                # 保存
                cv2.imwrite(output_path, result)
    print(f"超解像結果が {output_path} に保存されました。")

def overlay(base_path):
    img_path = os.path.join(base_path,'image')
    label_path = os.path.join(base_path,'label')
    save_path = os.path.join(base_path,'overlay')
    os.makedirs(save_path,exist_ok=True)
    print(img_path)
    imgs = [os.path.join(img_path, file) for file in os.listdir(img_path) if file.endswith((".png", ".tif", "tiff"))]
    img_path = sorted(imgs, key=lambda s: int(re.search(r'(\d+)\.', s).groups()[0]))
    labels = [os.path.join(label_path, file) for file in os.listdir(label_path) if file.endswith((".png", ".tif", "tiff"))]
    label_path = sorted(labels, key=lambda s: int(re.search(r'(\d+)\.', s).groups()[0]))
    
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in img_path]
    labels = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in label_path]

    for i in range(len(images)):
        overlap_image = gsh.overlay_img(raw_img=images[i],proce_img=labels[i], num= [0.8,0.2,0])
        cv2.imwrite(os.path.join(save_path,str(f'{i:05d}')+'.png'),overlap_image)



def main():
        
    # 使用例
    base_folder = "/home/nakajima/work/Ecoli/code/Unet/edsrdata/64to256"  # 現在の作業ディレクトリ
    name = "train"    # 新しいフォルダ名のベース
    model_path = "./EDSR_x4.pb"  # モデルファイルのパス
    scale = 4                  # スケール倍率
    source_folder = "/home/nakajima/work/Ecoli/code/Unet/data/64_16/"+ name + "_manual"  # 元画像が格納されたフォルダ

    # create_edsr_superres_folder(base_folder, name, model_path, scale, source_folder)

    base_path = '/home/nakajima/work/Ecoli/code/Unet/edsrdata/64to256/test'
    overlay(base_path)


if __name__ == "__main__":
    main()