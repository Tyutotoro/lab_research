import cv2
from PIL import Image
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from copy import copy
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
from PIL import ImageDraw

class Getshape:        
    #ディレクトリ作成
    def make_dir(self, save_path):
        pross_img_path = save_path + 'processing_/'
        overlay_img_path = save_path + 'overlay_/'
        os.makedirs(pross_img_path,exist_ok = True)
        os.makedirs(overlay_img_path,exist_ok = True)
        return pross_img_path,overlay_img_path
            

    # 画像差分を保存
    def img_diff(self, filelist,filelist2,save_path):
        img1 = cv2.imread(filelist)
        img2 = cv2.imread(filelist2)
        img_diff = cv2.absdiff(img1,img2)
        cv2.imwrite(f'{save_path}.png',img_diff)

    # フィルタ処理
    def imgfilter(self, img, filter_name: str, para):
        if filter_name == 'gau':
            return cv2.GaussianBlur(img, (5,5), 1,1)
        elif filter_name == 'bil':
            return cv2.bilateralFilter(img, d = 3,sigmaColor = 30, sigmaSpace = 10)
        elif filter_name == 'med':
            return cv2.medianBlur(img,ksize = 5)
        elif filter_name == 'mean':
            return cv2.blur(img, (5, 5))
        else:
            print("chose filter_name")

    #エッジ検出    
    def edge(self, img, edge_name, para):
        if edge_name == 'lap':
            return cv2.Laplacian(img, cv2.CV_8U, ksize = para[2])
        elif edge_name == 'sob':
            return cv2.Sobel(img,cv2.CV_8U, para[0], para[1], ksize = para[2])
        elif edge_name == 'ent':
            return entropy(img,disk(4))  
        else:
            print("chose edge_name")
        
    #二値化
    def threshold(self, img, thre_name):
        if thre_name == 'otsu':
            return self.Otsumethed(img)
        elif thre_name == 'bin':
            return self.binarization(img)
        else:
            print("chose thre_name") 
        
    #斜め微分
    def dia_filter(self, gray):
        kernel =  1 / 3 * np.array([[0, -1, 0], [-1, 0, 1], [0, 1, 0]])
        dst = cv2.filter2D(gray, -1, kernel)
        return dst

    #横微分
    def sq_filter(self, gray):
        # kernel = 1 / 3 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        dst = cv2.filter2D(gray, -1, kernel)
        return  dst
    #縦微分
    def ve_filter(self, gray):
        kernel = 1 / 3 * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        dst = cv2.filter2D(gray, -1, kernel)
        return  dst

    #カラー画像表示
    def color_img(self, src, type=str):
        if type == 'HOT':
            color_type = cv2.COLORMAP_HOT
        elif type == 'JET':
            color_type = cv2.COLORMAP_JET
        else:
            print('chose color type. by def color_img')
        color = cv2.applyColorMap(src, color_type)
        return color

    #閾値手動決定二値化
    def binarization(self, gray,num):
        ret, binary = cv2.threshold(gray, num, 255, cv2.THRESH_BINARY)
        print(ret)
        return binary

    #膨張処理
    def erode(self, img, kernel_n, n):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_n,kernel_n))
        img = cv2.erode(img, kernel)
        return img

    #収縮処理
    def dilate(self, img, kernel_n, n):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_n,kernel_n))
        img = cv2.dilate(img, kernel)
        return img

    # 画像のオーバーレイ
    def overlay_img(self,raw_img, proce_img,num):
        #raw_imgの重みnum[0], proce_imgの重みnum[1], num[2]はガンマ補正 
        overlay = cv2.addWeighted(raw_img, num[0], proce_img, num[1], num[2])
        return overlay
    #大津法
    def Otsumethed(self, data):
        r, dst = cv2.threshold(data, 0, 255, cv2.THRESH_OTSU)
        print(r)
        return dst

    #テンプレートマッチング
    def template_matching(self, temp, img, thres):
        # 処理対象画像に対して、テンプレート画像との類似度を算出する
        res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)

        # 類似度の高い部分を検出する
        threshold = thres
        loc = np.where(res >= threshold)

        # テンプレートマッチング画像の高さ、幅を取得する
        h, w = temp.shape

        # 検出した部分に赤枠をつける
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255), 2)
        return img

    #画像分割
    def split(self, img, sizev, sizeh):
        # print(img.shape[0],img.shape[1])
        v_size = img.shape[0] // sizev * sizev
        h_size = img.shape[1] // sizeh * sizeh
        # print(v_size,h_size)
        img = img[:v_size, :h_size]

        v_split = img.shape[0] // sizev
        h_split = img.shape[1] // sizeh
        out_img = []
        # print(v_split, h_split)
        [out_img.extend(np.hsplit(h_img, h_split))
        for h_img in np.vsplit(img, v_split)]
        return out_img

    # ハイパスフィルタ
    def highpath(self, f_xy,num,name,i):

        f_uv = np.fft.fft2(f_xy)
        # 画像の中心に低周波数の成分がくるように並べかえる
        shifted_f_uv = np.fft.fftshift(f_uv)

        # フィルタ (ハイパス) を用意する
        x_pass_filter = Image.new(mode='L',  # 8-bit pixels, black and white
                                size=(shifted_f_uv.shape[0],
                                        shifted_f_uv.shape[1]),
                                color=255,  # default white
                                )
        # 中心に円を描く
        draw = ImageDraw.Draw(x_pass_filter)
        # 円の半径
        ellipse_r = num
        # 画像の中心
        center = (shifted_f_uv.shape[0] // 2,
                shifted_f_uv.shape[1] // 2)
        # 円の座標
        ellipse_pos = (center[0] - ellipse_r,
                    center[1] - ellipse_r,
                    center[0] + ellipse_r,
                    center[1] + ellipse_r)
        draw.ellipse(ellipse_pos, fill=0)
        # フィルタ
        filter_array = np.asarray(x_pass_filter)

        # フィルタを適用する
        filtered_f_uv = np.multiply(shifted_f_uv, filter_array)

        # パワースペクトルに変換する
        magnitude_spectrum2d = 20 * np.log(np.absolute(filtered_f_uv))

        # 元の並びに直す
        unshifted_f_uv = np.fft.fftshift(filtered_f_uv)
        # 2 次元逆高速フーリエ変換で空間領域の情報に戻す
        i_f_xy = np.fft.ifft2(unshifted_f_uv).real  # 実数部だけ使う

        # 上記を画像として可視化する
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        # 枠線と目盛りを消す
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        # 元画像
        axes[0].imshow(f_xy, cmap='gray')
        axes[0].set_title('Input Image')
        # フィルタ画像
        # cv2.imwrite(f'{name}filter_array_{i}.png',filter_array)
        axes[1].imshow(filter_array, cmap='gray')
        axes[1].set_title('Filter Image')
        # フィルタされた周波数領域のパワースペクトル
        # cv2.imwrite(f'{name}magnitude_spectrum2d_{i}.png',magnitude_spectrum2d)
        axes[2].imshow(magnitude_spectrum2d, cmap='gray')
        axes[2].set_title('Filtered Magnitude Spectrum')
        # FFT -> Band-pass Filter -> IFFT した画像
        # cv2.imwrite(f'{name}i_f_xy_{i}.png',i_f_xy)
        axes[3].imshow(i_f_xy, cmap='gray')
        axes[3].set_title('Reversed Image')
        # グラフを表示する
        # plt.savefig(f'{name}{i}.png')
        return i_f_xy
        plt.show()

    # opencvのwatershed法
    def watershed_method(self, image):
        kernel = np.ones((3,3),np.uint8)
        # モルフォロジー演算のDilationを使う
        sure_bg = cv2.dilate(image,kernel,iterations=2)
        dist_transform = cv2.distanceTransform(image,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # foregroundの1オブジェクトごとにラベル（番号）を振っていく
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0
        # image = np.broadcast_to(image,(image.shape[0],image.shape[1],3))
        image = np.array([image,image,image])
        image = image.transpose(1,2,0)
        water_image = cv2.watershed(image,markers)    
        return water_image

    def make_mask(self,img, R, inv=False):
        """円形のマスク画像を作ります"""
        height = img.shape[0]
        width  = img.shape[1]

        center_w = height//2
        center_h = width//2

        if inv:
            n = 0
            filter_matrix = np.ones([height, width])
        else:
            n = 1
            filter_matrix = np.zeros([height, width])

        for i in range(0, height):
            for j in range(0, width):
                    if (i-center_w)*(i-center_w) + (j-center_h)*(j-center_h) < R*R:
                                filter_matrix[i][j] = n

        return filter_matrix

    def masked_fft(self,img, mask):
        dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)*mask

        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        magnitude_spectrum[magnitude_spectrum==-np.inf]=0

        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

        dft2 = cv2.dft(np.float32(img_back),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift2 = np.fft.fftshift(dft2)

        magnitude_spectrum2 = 20*np.log(cv2.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))

        return magnitude_spectrum, img_back, magnitude_spectrum2


    def highpath2(self, img_gray,name,j):

        masks_i = np.array([np.array([self.make_mask(img_gray, r, True), self.make_mask(img_gray, r, True)]).transpose(1, 2, 0) for r in [10, 20, 30]])

        # fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        # for i in range(3):
        #     spectrum_img1, img_ifft, spectrum_img2 = self.masked_fft(img_gray, masks_i[i])
            # ax[0][i].imshow(spectrum_img1, 'gray')
            # ax[0][i].set_xticks([])
            # ax[0][i].set_yticks([])

            # ax[1][i].imshow(img_ifft, 'gray')
            # ax[1][i].set_xticks([])
            # ax[1][i].set_yticks([])
        # plt.savefig(f'{name}102040_{j}.png')
        spectrum_img1, img_ifft, spectrum_img2 = self.masked_fft(img_gray, masks_i[0])
        # print(img_ifft)
        # print(img_ifft.dtype)
        # print(type(img_ifft))
        # print(img_ifft.shape)
        # cv2.imwrite(f'{name}r30_{j}.tif', img_ifft)
        return img_ifft
        # plt.show()



    
    #ヒストグラム表示
    def hist(self, data,name):
        plt.hist(data.flatten(), bins=np.arange(256))
        plt.savefig(f'{name}_hist.png')
        plt.show
        plt.clf()


