import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage
import copy
import inspect
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.ndimage import center_of_mass

from skimage import filters, morphology, measure
from scipy.spatial import ConvexHull
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from collections import deque
# from .gmm_mml.gmm_mml_module.gmm_mml2 import GmmMml2
import csv
import networkx as nx
import collections
# from gmm_mml.gmm_mml import GmmMml
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from picture import Picture
from make_graph import make_length_graph
# from BaSCA2 import cal_lengh_main, save_to_csv

def cal_lengh_thinning(image,save_path):
    """
    画像内の各クラス領域に対して長さを測定するmain関数
    """
    length_list = []  # 長さリスト
    back_image = np.zeros_like(image)
    for num in range(1, np.max(image) + 1):
        # 値に対応する領域を抽出
        region = (image == num)
        if not np.any(region):
            continue
        #等高線の作成
        concat_array = find_contour_line(region)
        
        concat_array = np.where(concat_array > 0, 255, 0).astype(np.uint8)
        skeleton2 =  cv2.ximgproc.thinning(concat_array, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        #長さの測定
        length = cv2.countNonZero(skeleton2)
        node_coord = np.column_stack(np.where(skeleton2 == 255))
        back_image = make_length_image(back_image,region,concat_array,node_coord)
        # print(length, node_coord)
        # plt.imshow(back_image)
        # plt.show()
        # 
        length_list.append([num, length])
    #長さリストをヒストグラムで表し保存
    if length_list:
        values = [entry[1] for entry in length_list if entry[1] is not None]
        # ヒストグラムを作成
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=np.arange(min(values), max(values) + 2) - 0.5, edgecolor='black')
        plt.xlim(right = 100)
        plt.ylim(top = 350)
        plt.xlabel("cell length")
        plt.ylabel("Count")
        plt.title('Histogram of cell length')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show
        plt.savefig(save_path + "/cal_cell_histogram_thinning.png")
    return length_list, values, back_image




def cal_lengh_main(image,save_path):
    """
    画像内の各クラス領域に対して長さを測定するmain関数
    """
    length_list = []  # 長さリスト
    back_image = np.zeros_like(image)
    for num in range(1, np.max(image) + 1):
        # 値に対応する領域を抽出
        region = (image == num)
        if not np.any(region):
            continue
        #等高線の作成
        
        concat_array = find_contour_line(region)
        #長さの測定
        length,node_coord = cal_cell_length(concat_array)
        back_image = make_length_image(back_image,region,concat_array,node_coord)
        # print(length, node_coord)
        # plt.imshow(back_image)
        # plt.show()
        # 
        length_list.append([num, length])
    #長さリストをヒストグラムで表し保存
    if length_list:
        values = [entry[1] for entry in length_list if entry[1] is not None]
        # ヒストグラムを作成
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=np.arange(min(values), max(values) + 2) - 0.5, edgecolor='black')
        plt.xlim(right = 100)
        plt.ylim(top = 350)
        plt.xlabel("cell length")
        plt.ylabel("Count")
        plt.title('Histogram of cell length')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show
        plt.savefig(save_path + "/cal_cell_histogram2.png")
    return length_list, values, back_image

def find_contour_line(img):
    """
    輪郭線の探索,等高線の作成
    """
    #抽出した領域のひと回り大きい配列を作成
    copyimg  = copy.deepcopy(img)
    coords = np.argwhere(copyimg)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    padded_shape = tuple(max_coords - min_coords + 2)
    pad_array = np.zeros(padded_shape, dtype=np.uint8)
    region_offset = coords - min_coords + 1
    for coord in region_offset:
        pad_array[tuple(coord)] = 1
    #抽出した領域を境界線から中心に向かってランク付け
    num=1
    back_img = np.zeros_like(pad_array)
    padded_array = copy.deepcopy(pad_array)
    while True:
        reversed_binary_image = np.max(padded_array) - padded_array     
        shrink_image= ndimage.binary_dilation(reversed_binary_image)
        border_pixels = np.logical_and(shrink_image, padded_array == 1)

        contour_image = np.where(border_pixels>0,num,0)
        back_img = contour_image + back_img
        padded_array = padded_array- border_pixels
        num +=1
        if np.all(padded_array == 0):
            break
    # return back_img
    return pad_array


def cal_cell_length(image):
    """
    グラフの作成、長さの測定、長さの採用の判定のmain関数
    """
    graph = nx.Graph()
    graph = make_graph(image,graph)
    # pos = nx.spring_layout(graph, seed=42)
    # nx.draw(graph) #グラフの描画(おまかせ)
    # edge_labels = nx.get_edge_attributes(graph, 'weight')
    # print(edge_labels)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=12)
    # plt.show() #グラフの描画
    start_end_points = find_start_and_end_nodes(image, graph)  
    shortpath_list = []
    for i in range(len(start_end_points)):
        for j in range(i + 1, len(start_end_points)):  
            short_path = find_path(graph,start_end_points[i], start_end_points[j])
            if not np.any(np.isnan(np.array(short_path))):
                shortpath_list.append([len(short_path),list(short_path)])
    try:
        if shortpath_list:
            max_length_index = np.argmax(np.array(shortpath_list)[:, 0])
            # 最長経路に含まれるノードIDを取得
            max_length_nodes = shortpath_list[max_length_index][1]  # ノードIDの配列
            max_length = shortpath_list[max_length_index][0] 
            # ノードIDに対応する座標を取得
            node_coordinates = [graph.nodes[node]['pos'] for node in max_length_nodes]
            # max_length = np.max(np.array(shortpath_list)[:,0])
            if max_length == 1:
                print('length is 1')
                print(max_length, node_coordinates)
            return max_length, node_coordinates
        else:
            # print(short_path)
            print(graph)
            print(list(nx.get_node_attributes(graph, 'pos')[0]))
            print('error')
            return 1,list(nx.get_node_attributes(graph, 'pos')[0])
    except:
        # print(short_path)
        print(graph)
        print(list(nx.get_node_attributes(graph, 'pos')[0]))
        print('None')
        return 0, list(nx.get_node_attributes(graph, 'pos')[0])

def make_graph(image,graph):
    """
    画像からグラフを作成する
    """
    # graph = nx.Graph()
    rows, cols = image.shape

    # 8近傍の移動量（y, x）
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    pixel_to_node = {}
    node_id = 0

    # ピクセル探索してノード、エッジを登録
    for y in range(rows):
        for x in range(cols):
            if image[y, x] > 0:  # 値が0より大きい場合に対象とする
                if (y, x) not in pixel_to_node:
                    pixel_to_node[(y, x)] = node_id
                    # graph.add_node(node_id, pos=(y, x), weight=1/image[y, x])  # ノード追加と重み設定
                    graph.add_node(node_id, pos=(y, x), weight=1)  # ノード追加と重み設定
                    node_id += 1

                current_node = pixel_to_node[(y, x)]

                # 8近傍の探索
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and image[ny, nx] > 0:
                        if (ny, nx) not in pixel_to_node:
                            pixel_to_node[(ny, nx)] = node_id
                            # graph.add_node(node_id, pos=(ny, nx), weight=1/image[ny, nx])  # ノード追加と重み設定
                            graph.add_node(node_id, pos=(ny, nx), weight=1)  # ノード追加と重み設定
                            node_id += 1

                        neighbor_node = pixel_to_node[(ny, nx)]
                        if not graph.has_edge(current_node,neighbor_node):
                            # edge_weight = ((1/image[y, x]) + (1/image[ny, nx]))  # エッジの重みをノード値で設定
                            edge_weight = 1  # エッジの重みをノード値で設定
                            graph.add_edge(current_node, neighbor_node, weight=edge_weight)
    return graph

def find_start_and_end_nodes(image, graph):
    """
    スタートとゴールのポイントを凸包から探す
    """
    rows, cols = image.shape
    # グラフからノードの座標を取得
    pos_to_node = {data['pos']: node for node, data in graph.nodes(data=True)}

    # 輪郭線の座標を格納するためのリスト
    contours = []
    # 輪郭を検出
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 1:  # 値が1の場合
                contours.append((i, j))  # 配列形式で座標を格納
    contours = np.array(contours)
    # 各輪郭の凸包を計算
    hull = cv2.convexHull(contours)  # 凸包の計算
    hull_sq = np.squeeze(hull)
    start_end_points = []
    if hull_sq.ndim == 2:    
        for hull_point in hull_sq:
            point = pos_to_node.get((hull_point[0], hull_point[1]), None)
            start_end_points.append(point)
    else:
        start_end_points.append(1)
    return start_end_points
    
def find_path(graph,start_node,end_node):
    """
    最短経路の探索
    """
    try:
        short_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight', method='dijkstra')
        return short_path
    except nx.NetworkXNoPath:
        return np.nan

def save_to_csv(data, filename, headers=None):
    """
    データをCSVファイルに保存する関数。
    """

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
    print('save csv file')

def make_length_image(back_image, region, concat_array,length_path):
    #座標の取得
    copyimg  = copy.deepcopy(region)
    coords = np.argwhere(copyimg)
    min_coords = coords.min(axis=0)
    # print(min_coords)
    #細胞長の経路埋め込み
    concat_array_bg = np.zeros_like(concat_array)
    if len(np.array(length_path).shape) == 2:
        for coord in length_path:
            row, col = coord
            concat_array_bg[row, col] = 1  # 指定座標を1に更新
    else:
        concat_array_bg[length_path[0],length_path[1]] = 1
    #周囲の余分な箇所を削除
    concat_array_bg_removed = concat_array_bg[1:-1, 1:-1]
    #元の画像の座標に加算
    start_row, start_col = min_coords
    end_row, end_col = start_row + concat_array_bg_removed.shape[0], start_col + concat_array_bg_removed.shape[1]
    back_image[start_row:end_row, start_col:end_col] += concat_array_bg_removed
    return back_image

def make_overlap_image(truelabel, cover_image,base_path):
    #overlapするときの関数
    #白色が正解ラベル
    #青色が予測結果
    true_label = np.where(truelabel == 0, 200, truelabel)
    background = np.zeros_like((cover_image), dtype=np.uint8)
    cover_image_color = np.array([background,background,cover_image]).transpose(1,2,0)
    true_label_color = np.array([true_label, true_label,true_label]).transpose(1,2,0)
    overlap_image = cv2.addWeighted(true_label_color,0.8,cover_image_color,0.2,0 )
    overlap_image_rgb = overlap_image[:, :, [2, 1, 0]]
    overlap_image_rgb = np.where((overlap_image_rgb == [211,160,160]).all(axis=-1)[..., None],[160,255,255] ,overlap_image_rgb)#外れてる箇所
    overlap_image_rgb= np.where((overlap_image_rgb ==  [255,204,204]).all(axis=-1)[..., None], [0,0,255], overlap_image_rgb)#当たっている箇所
    overlap_image_rgb= np.where(overlap_image_rgb == 204, 255, overlap_image_rgb)#正解ラベルのみの箇所
    overlap_image_bgr = overlap_image_rgb[:, :, [2, 1, 0]]
    # true_label_color = np.array([truelabel, truelabel,truelabel]).transpose(1,2,0)
    # background = np.zeros_like((cover_image), dtype=np.uint8)
    # cover_image_color = np.array([background,background,cover_image]).transpose(1,2,0)
    # overlap_image = cv2.addWeighted(true_label_color,0.2,cover_image_color,0.8,0 )
    cv2.imwrite(os.path.join(base_path,'overlap_length_image_thinning.png'),overlap_image_bgr)
    cv2.imwrite(os.path.join(base_path,'length_image_thinning.png'),cover_image)

def count_lengh_cal(image,save_path):
    np.random.seed(3407)
    normal_image = np.where(image > 0, 255, image)
    length_list, values , back_image= cal_lengh_thinning(image,save_path)
    back_image = np.where(back_image > 0, 255, back_image)
    # CSVファイルに保存
    make_overlap_image(normal_image,back_image,save_path)
    # save_to_csv(length_list, save_path + "/length_list_gmmmml_thinning.csv", headers=["class_number","Row Count"])
    # make_length_graph(save_path, "/length_list_gmmmml_thinning.csv",True)

def main(input_name):
    # input_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1'
    # input_path = '/home/nakajima/work/Ecoli/code/Unet/result'
    input_path = '/home/nakajima/work/Ecoli/code/BaSCA'
    inputs_path = os.path.join(input_path,input_name)
    file_name = 'gmmmml_np_savetxt3.csv'
    csv_data = np.genfromtxt(os.path.join(inputs_path,file_name), delimiter=',')
    csv_data = csv_data.astype(np.int64)
    print(csv_data.shape)
    count_lengh_cal(csv_data, inputs_path)
    # print(type(data))
    # print(np.unique(data))
if __name__ == "__main__":
    name_list = [
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

        'manual_scalelabel_ver2',
        # 'manual_label_ver3',
    ]
    for name in name_list:
        main(name)