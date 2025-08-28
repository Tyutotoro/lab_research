import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def save_all_frames(video_path, dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            # cal_image(frame)
            # plt.imshow(frame)
            # plt.show()
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return
        
def read_image(path):
    region_list = []
    region_ratio_list = []
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            img_path = os.path.join(path, filename)
            # image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            region =cal_image(image)
            all_pix = image.shape[0]*image.shape[1]
            region_ratio = region/all_pix
            # print(region_ratio)
            # if region_ratio < 0.1:
            #     plt.imshow(image)
            #     plt.show()
            region_list.append(region)
            region_ratio_list.append(region_ratio*100)
    
    plt.figure(figsize=(20, 12))
    plt.violinplot(region_ratio_list,showmeans=True,showmedians=True)
    plt.show()
    # print(region_list)


def cal_image(frame):
    class_num = np.unique(frame)
    frame_thel = np.where(frame>0 ,255,0)
    # if not np.all(frame_thel == 0):
    region = cv2.countNonZero(frame_thel)
    # else:
        # region = 0
    # 2つ並べて表示
    # plt.figure(figsize=(10, 5))  # 画像サイズの調整

    # plt.subplot(1, 2, 1)
    # plt.imshow(frame)
    # plt.title("Image 1")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(frame_thel)
    # plt.title("Image 2")
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()
    # plt.imshow(frame)
    # plt.show()
    # plt.imshow(frame_thel)
    # plt.show()
    # print(region)
    return region
#ぴくせる数、各クラスごとのぴくせる数、比率、画像全体での比率


def main():
    movie_path = "/home/nakajima/work/Ecoli/code/stardist/TRAgen_data/stardist_fluo_colourMask.avi"
    # save_path = "/home/nakajima/work/Ecoli/code/stardist/TRAgen_data/label_image"
    # save_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/scaledata/256_16/test_manual2/label'
    save_path = '/home/nakajima/Downloads/dsb2018/test/masks'
    # save_all_frames(movie_path, save_path, 'sample_video_img')
    read_image(save_path)

# save_all_frames('data/temp/sample_video.mp4', 'data/temp/result_png', 'sample_video_img', 'png')

if __name__ == "__main__":
    main()