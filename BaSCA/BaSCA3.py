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
from .gmm_mml.gmm_mml_module.gmm_mml2 import GmmMml2
# from .gmm_mml.gmm_mml_module.gmm_mml3 import GmmMml3

import csv
import networkx as nx
import collections
# from gmm_mml.gmm_mml import GmmMml
from scipy.stats import mode
import sys
import os

from matplotlib.colors import ListedColormap,LinearSegmentedColormap
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from picture import Picture

#ここからBaSCAのbowtieを分割する関数など
#--------------------------------------------#
# #スケルトン化画像から個体ごとにラベル付け
def find_connected_components(image):
    # 8近傍で連結成分をラベル付け
    skeleton = image_skeletonize(image)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton)
    # クラスごとに255の座標を取得
    groups = []
    for i in range(1, n_labels):
        coords = np.column_stack(np.where(labels == i))
        groups.append(coords)
    return groups

#各ピクセルごとの背景までの距離計算
def compute_nearest_distance(groups, original_image):
    distance_map = distance_transform_edt(original_image != 0)
    total_distances = []
    for group in groups:
        distances = []
        for i in group:
            min_distance = distance_map[[i[0]],[i[1]]]
            distances.append(min_distance[0])
        distances = np.array(distances)
        total_distances.append(distances)
    return total_distances

#各クラスごとに分類
def split_distance(distances,groups,image,limit=0):
    d = 0
    t = 0.7 #0.65 ~ 0.75の間
    peaks, _ = find_peaks(distances)
    valleys, _ = find_peaks(-distances)
    # 条件を満たす最小値を選択
    selected_valleys = []
    split_points = []
    for valley in valleys:
        leftlmax = max(distances[peaks[peaks < valley]]) if np.any(peaks < valley) else None
        rightlmax = max(distances[peaks[peaks > valley]]) if np.any(peaks > valley) else None
        # 条件を満たすか確認
        if leftlmax is not None and rightlmax is not None:
            if (distances[valley] / leftlmax <= t) and (distances[valley] / rightlmax <= t):
                selected_valleys.append(valley)
                split_points.append([groups[valley-1], groups[valley],groups[valley+1]])

    image_color = np.array([image,image,image])
    image_color = image_color.transpose(1, 2, 0)
    #領域の分割
    #peak値を基準にスケルトンの線の法線ベクトルを計算
    if len(split_points):
        change_pixels = []
        for i,point in enumerate(split_points):
            back = point[0]
            centor = point[1]
            front = point[2]
            vector_x = -(front[1]-back[1])
            vector_y = front[0]-back[0]
            norm_length = np.sqrt(vector_x**2 + vector_y**2)
            #法線ベクトル
            vector_y = vector_y/norm_length
            vector_x = vector_x/norm_length
            #法線ベクトルに沿って背景にあたるまで0にする
            #法線ベクトルの正の向きに沿って動く
            d_forward = 0
            while True:
                ny = int(round(centor[0] + d_forward * vector_y))
                nx = int(round(centor[1] + d_forward * vector_x))
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    if image[ny, nx] != 0:
                        change_pixels.append([ny, nx])
                        image[ny, nx] = 0
                    else:
                        break
                else:
                    break
                d_forward += 1
            #法線ベクトルの負の方向に沿って動く
            d_backward = 0
            while True:
                ny = int(round(centor[0] - d_backward * vector_y))
                nx = int(round(centor[1] - d_backward * vector_x))
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    if image[ny, nx] != 0:
                        change_pixels.append([ny, nx])
                        image[ny, nx] = 0
                    else:
                        break
                else:
                    break
                d_backward += 1
        for i, pixel in enumerate(change_pixels):
            image_color[pixel[0],pixel[1]] = [0,0,255]
    return peaks,selected_valleys,image_color,image
#--------------------------------------------#


#ここからBaSCAのwatershedやgmmを使って分割する関数など
#-------------------------------------------------#
#watershed法
def apply_watershed(image):
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)),labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_maxi.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels, distance

#ソリッド度による結合判定
def merge_solid_fragments(labels, integration_list):
    new_labels = copy.deepcopy(labels)
    max_solidity = 0.0
    change_num = 0
    for int_label in integration_list:
        coords = np.argwhere(int_label >= 1)
        if coords.shape[0] < 3:
            print('凸包できない')
            return 0  # 凸包が定義できないほど少ないなら solidity = 0
        try:
            hull = ConvexHull(coords)
            hull_area = hull.volume  # 2Dでは volume が面積に相当
        except:
            return 0  # 共線など凸包構築失敗時は安全に0と返す
        region_area = coords.shape[0]  # ピクセル数
        solidity = region_area / hull_area
        if solidity >= max_solidity:
            change_num = np.max(int_label)
            min_num = np.min(int_label[int_label > 0])
            max_solidity = solidity
    if max_solidity != 0:
        new_labels[new_labels == change_num] = min_num
    return new_labels

def get_neighbors(y, x, shape):
    """
    8近傍の座標を返す
    """
    neighbors = []
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1), (1, 0),  (1, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
            neighbors.append((ny, nx))
    return neighbors

def merge_region(labels, region_list):
    integration_list =[]
    for region in region_list:
        region_normal = copy.deepcopy(region)
        #スケルトン化
        region_normal[region_normal >= 1] = 1
        region_normal = region_normal.astype(np.uint8)
        skeleton_region = image_skeletonize(region_normal)
        #深い谷を適応
        #スケルトンを距離マップに適応
        groups =  np.argwhere(skeleton_region != 0)
        distances = compute_nearest_distance([groups], region_normal)
        t = 0.7 #0.65 ~ 0.75の間
        distances = np.squeeze(distances)
        peaks, _ = find_peaks(distances)
        valleys, _ = find_peaks(-distances)
        # 条件を満たす最小値を選択
        selected_valleys = []
        for valley in valleys:
            leftlmax = max(distances[peaks[peaks < valley]]) if np.any(peaks < valley) else None
            rightlmax = max(distances[peaks[peaks > valley]]) if np.any(peaks > valley) else None
            # 条件を満たすか確認
            if leftlmax is not None and rightlmax is not None:
                if (distances[valley] / leftlmax <= t) and (distances[valley] / rightlmax <= t):
                    selected_valleys.append(valley)
        #深い谷が無ければ統合候補に追加
        if len(selected_valleys) == 0:
            integration_list.append(region)
    integration_list = np.array(integration_list)
    #統合候補が１つであればそのまま、2つ以上であればソリッド度で判定
    if len(integration_list) == 1:
        labels[labels == np.max(integration_list)] = np.min(integration_list[integration_list > 0])
    elif len(integration_list) >= 2:
        labels = merge_solid_fragments(labels,integration_list)
    else:
        None
    return labels

def puzzle_solving(labels):
    """
    1.watershedのクラスごとの重心を求める
    細線化する、深い谷に入れる
    2.「深い谷」アルゴリズムを再度適用
    2-1.隣接しているクラスとの深い谷アルゴリズムで谷がなければクラスの統合候補
    2-2.他の隣接しているクラスで同様に行う。これを繰り返す
    2-3.solidityの判定もいれる
    3.再帰終わりに返す
    """
    labels = copy.deepcopy(labels)
    processed_min_vals = set()
    while True:
        region_list = []
        # ステップ1: 0以外＆未処理から最小値を探索
        unique_vals = np.unique(labels)
        candidate_vals = [v for v in unique_vals if v != 0 and v not in processed_min_vals]
        if not candidate_vals:
            break

        current_min = min(candidate_vals)
        processed_min_vals.add(current_min)

        # ステップ2: current_min の隣接値を収集
        adjacent_vals = set()
        positions = np.argwhere(labels == current_min)

        for y, x in positions:
            for ny, nx in get_neighbors(y, x, labels.shape):
                neighbor_val = labels[ny, nx]
                if neighbor_val != current_min and neighbor_val != 0:
                    adjacent_vals.add(neighbor_val)

        # ステップ3-2: current_min + 各異なる値について処理
        for val in adjacent_vals:
            mask = (labels == current_min) | (labels == val)
            region = np.where(mask, labels, 0)
            region_list.append(region)

        labels = merge_region(labels, region_list)
    return labels

#ガウス分布に基づくデータ生成
def generate_detapoints(labels, distance):
    #パイパラ
    variance=0.3

    """
    全体の面積のパターン
    """
    generate_num = (np.count_nonzero(labels))

    """
    全体の面積から1個体の面積算出パターン
    """
    # unique_values, counts = np.unique(labels[labels != 0], return_counts=True)
    # mediann_erea = np.median(counts)
    # generate_num = len(unique_values)*mediann_erea

    """
    論文のパターン
    """
    # CellLength = 3.5 
    # CellWidth = 1.1
    # CalFactor = 0.12
    # generate_num = (np.unique(labels)-1)*((CellLength * CellWidth)/CalFactor**2)/2

    object_pixels = np.column_stack(np.nonzero(labels))
    distances = np.array([distance[x, y] for x, y in object_pixels])

    if distances.sum() > 0:
        weights_pi = distances / distances.sum()
    else:
        # 重みがゼロの場合は全ての非ゼロピクセルに等しい重みを与える
        weights_pi = np.ones_like(distances) / len(distances)
    if np.max(generate_num * weights_pi) < 1:
        weights_pi = np.full(weights_pi.shape,2)
    # 各ピクセルからデータポイントを生成
    data_points = []
    for coord, weight in zip(object_pixels, weights_pi):
        num_points = int(generate_num * weight) 
        x, y = coord
        # ガウス分布に基づくデータポイントを生成
        points_x = np.random.normal(x, variance, num_points)
        points_y = np.random.normal(y, variance, num_points)
        data_points.append(np.column_stack((points_x, points_y)))
            # データポイントが空でない場合のみ結合
    if data_points:
        return np.vstack(data_points)
    else:
        # 全ての非ゼロピクセルをデータポイントにする
        return object_pixels    

#gmmのmmlを用いたモデル
def gmmmml2(coords, n_components):
    unsupervised = GmmMml2(kmin=1,
                kmax=n_components,
                regularize=1e-6,
                threshold=1e-5,
                covoption=0,
                max_iters=100,
                plots=True)
    unsupervised = unsupervised.fit(coords,verb=True)
    classified = unsupervised.classify_components(coords)
    return unsupervised, classified

def apply_clustering_to_image(labeled_image, label_id, clus_model):
    # クラスタリング結果を元の画像に適用
    coords = np.column_stack(np.where(labeled_image == label_id))
    cluster_labels = clus_model.predict(coords)
    # cluster_labels = clus_model.fit_predict(coords)
    result = np.zeros_like(labeled_image, dtype=np.int32)
    for coord, label in zip(coords, cluster_labels):
        result[tuple(coord)] = label + 1  # ラベルは1から始める
    return result
#--------------------------------------------#

#細胞領域の分岐判定
def count_branches_and_paths(binary_image):
    def is_within_bounds(x, y, image):
        return 0 <= x < image.shape[0] and 0 <= y < image.shape[1]

    def get_neighbors(x, y):
        return [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)  # Vertical and horizontal neighbors
        ]

    def get_diagonal_neighbors(x, y):
        return [
            (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)  # Diagonal neighbors
        ]

    def explore_path(start, counter_ref):
        path = [start]
        x, y = start
        visited.add((x, y))
        sub_paths = []

        while True:
            vertical_neighbors = [(nx, ny) for nx, ny in get_neighbors(x, y) if is_within_bounds(nx, ny, binary_image) and binary_image[nx, ny] == 255 and (nx, ny) not in visited]
            diagonal_neighbors = [(nx, ny) for nx, ny in get_diagonal_neighbors(x, y) if is_within_bounds(nx, ny, binary_image) and binary_image[nx, ny] == 255 and (nx, ny) not in visited]

            all_neighbors = vertical_neighbors + diagonal_neighbors

            if len(all_neighbors) == 0:
                break

            if len(all_neighbors) == 1:
                x, y = all_neighbors[0]
                path.append((x, y))
                visited.add((x, y))

            elif len(all_neighbors) == 2:
                if abs(all_neighbors[0][0] - all_neighbors[1][0]) + abs(all_neighbors[0][1] - all_neighbors[1][1]) == 1:
                    x, y = [n for n in vertical_neighbors if n in all_neighbors][0]
                    path.append((x, y))
                    visited.add((x, y))
                else:
                    counter_ref[0] += 1
                    for neighbor in all_neighbors:
                        if neighbor not in visited:
                            sub_path = explore_path(neighbor, counter_ref)
                            sub_paths.extend(sub_path)
                    break

            else:
                counter_ref[0] += 1
                for neighbor in all_neighbors:
                    if neighbor not in visited:
                        sub_path = explore_path(neighbor, counter_ref)
                        sub_paths.extend(sub_path)
                break

        return [path] + sub_paths

    def process_image(counter_ref, paths_ref, traversal_order, visited):
        for i, j in traversal_order:
            if binary_image[i, j] == 255 and (i, j) not in visited:
                path = explore_path((i, j), counter_ref)
                if path:
                    paths_ref.append(path)

    # Define counters and paths
    counter1 = [0]
    counter2 = [0]
    paths1 = []
    paths2 = []

    # First traversal: Top-left to bottom-right
    visited = set()
    traversal_order1 = [(i, j) for i in range(binary_image.shape[0]) for j in range(binary_image.shape[1])]
    process_image(counter1, paths1, traversal_order1, visited)

    # Second traversal: Bottom-right to top-left (column-wise, bottom to top)
    visited = set()
    traversal_order2 = [(i, j) for j in range(binary_image.shape[1] - 1, -1, -1) for i in range(binary_image.shape[0] - 1, -1, -1)]
    process_image(counter2, paths2, traversal_order2, visited)

    # Compare counters and return the results
    if counter1[0] <= counter2[0]:
        return counter1[0], paths1
    else:
        return counter2[0], paths2

#画像読み込み
def load_image(path):
    try:
        input_image = tifffile.imread(path)
    except:
        input_image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return input_image

#画像の保存
def save_image(path,image):
    if np.max(image)>255:
        try:
            tifffile.imwrite(path,image.astype(np.uint16))
        except:
            cv2.imwrite(path,image.astype(np.uint16))
    else:
        try:
            tifffile.imwrite(path,image.astype(np.uint8))
        except:
            cv2.imwrite(path,image.astype(np.uint8))

#スケルトン化
def image_skeletonize(image):
    #正規化
    norm_image = cv2.normalize(image,None,0,1,cv2.NORM_MINMAX)
    skeleton   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton2   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    # skeleton1 = skeletonize(norm_image)
    return skeleton2  


#bascaのmain関数
def basca_main(image,image_number,save_path):
    #スケルトン画像の作成
    sk_image = image_skeletonize(image)
    
    #スケルトンのオブジェクトを分類
    branch_count, path_list = count_branches_and_paths(sk_image)
    if branch_count>0:
        result_image = watershed_basca_main(sk_image, image,image_number,save_path)
    else:
        result_image = brige_basca_main(sk_image,image,image_number)
    return result_image


#BaSCAの直線の方のmain関数
def brige_basca_main(sk_image, image,num,save_path = None):
    save_path = '/home/nakajima/work/Ecoli/code/BaSCA/manual_label'
    gro = find_connected_components(image)
    dis = compute_nearest_distance(gro,image)
    total_vall = 0
    for i, dist in enumerate(dis):
        peak, vall,image_color,image_gray = split_distance(dist,gro[i],image)
        if np.max(image_gray) == 255:
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_gray)
            image = labels
        total_vall = total_vall+ len(vall)
    return image

#BaSCAの枝分かれの方のmain関数
def watershed_basca_main(sk_image, image,image_number,save_path):
    #watershed法の適応
    labels, distance = apply_watershed(image)
    # print(f'bef{labels}')
    # print(distance)

    #ソリッド度、深い谷を用いたクラス分類の修正
    labels =puzzle_solving(labels)
    #クラス数のカウント
    if np.all(labels == 0):
        cluster_result = image
        # print('label 0')
    else: 
        n_components = measure.label(labels).max()
        #ガウス分布に従ってサンプルを抽出
        coords = generate_detapoints(labels,distance)
        if int(coords.shape[0]) > 10000:
            print(f'large coords num : {coords.shape}')
        #     max_points = int(coords.shape[0])//100  # 最大データポイント数
        #     coords = reduce_data_points(coords, max_points)
        #ガウス混合モデルでクラス分類の最適化
        if coords.shape[0] <=1:
            print("coords shape is 1 or 0")
            cluster_result = image
        else:
            clsmodel, k = gmmmml2(coords, n_components)
            cluster_result = apply_clustering_to_image(image, 255, clsmodel)
    return cluster_result

# データ削減関数
def reduce_data_points(data_points, max_total_points):
    if len(data_points) > max_total_points:
        indices = np.random.choice(len(data_points), max_total_points, replace=False)
        return data_points[indices]
    return data_points

def process_array(array, num):
    # 配列の0を除くユニークな値を検出
    unique_classes = np.unique(array[array != 0])
    # 非ゼロの値にnumを加算
    updated_array = np.where(array != 0, array + num, array)
    return unique_classes, updated_array

# 条件に基づいた画像スキャンと処理
def process_main(image):
    """
    このプログラムのメイン関数
    
    指定条件を満たす画像スキャンと処理を行う。
    引数
        image (ndarray): 入力画像(2値画像)
        
    返り値
        result_image (ndarray): 抽出された領域を含む画像。
    """
    # 結果を保存する黒色の画像
    result_image = np.zeros(image.shape)
    # 処理済みピクセルを追跡するためのマスク
    processed_mask = np.zeros_like(image, dtype=bool)
    
    # 抽出された領域を格納するリスト
    extracted_regions = []  # 各領域の画像を格納するリスト
    num = 0
    def find_connected_pixels(x, y, visited, boundary):
        """
        8近傍で白色のピクセルを探索し、境界を更新する。
        """
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            # 境界を更新
            boundary[0] = min(boundary[0], cx)  # 上
            boundary[1] = max(boundary[1], cx)  # 下
            boundary[2] = min(boundary[2], cy)  # 左
            boundary[3] = max(boundary[3], cy)  # 右
            
            # 8近傍を探索
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),         (0, 1),
                            (1, -1), (1, 0), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                    if image[nx, ny] == 255 and (nx, ny) not in visited:
                        stack.append((nx, ny))
    image_number = 0
    # スキャン
    for row in range(0, image.shape[0], 3):
        for col in range(0, image.shape[1], 3):
            if image[row, col] == 255 and not processed_mask[row, col]:
                # 白色ピクセルを発見したら処理開始
                visited = set()
                boundary = [row, row, col, col]  # 上, 下, 左, 右の境界
                
                # 連結領域を探索
                find_connected_pixels(row, col, visited, boundary)
                
                # 長方形領域を抽出
                top, bottom, left, right = boundary
                for vx, vy in visited:
                    processed_mask[vx, vy] = True  # 処理済みとしてマーク
                # 元画像から長方形領域を切り取る
                extracted_rectangle = image[top:bottom+1, left:right+1].copy()
                # リストに保存
                extracted_rectangle = np.pad(np.array(extracted_rectangle), pad_width=1, mode='constant', constant_values=0)
                #basca処理
                extracted_rectangle = basca_main(extracted_rectangle,image_number,'none')
                
                extracted_rectangle = extracted_rectangle[1:-1, 1:-1].astype(np.uint64)
                unique_classes, extracted_rectangle_class = process_array(extracted_rectangle, num)
                extracted_regions.append(extracted_rectangle_class)
                # 結果画像に安全に貼り付け
                mask = result_image[top:bottom+1, left:right+1] == 0
                result_image[top:bottom+1, left:right+1][mask] = extracted_rectangle_class[mask]  
                num = num + len(unique_classes)

                image_number+=1
    return result_image, extracted_regions

def compress_array(arr):
    # 1. 入力配列をフラットにして一意な値を取得
    flattened = arr.flatten()
    unique_values = np.unique(flattened)
    
    # 2. 存在しない値を検出
    full_range = np.arange(0, np.max(arr))  
    missing_values = np.setdiff1d(full_range, unique_values)  # 欠損している値
    
    # 3. 実際の値を 0 から順番に詰める
    value_map = {v: i for i, v in enumerate(unique_values)}  # 現在の値 → 詰めた値へのマッピング
    
    # 4. マッピングを使って値を変換
    compressed = np.array([value_map[v] for v in flattened])
    
    # 5. 2次元配列に戻す
    result = compressed.reshape(arr.shape)
    return result

"""
---------------------------
ここから下の関数は長さ計測の関数
--------------------------
"""
def cal_lengh_main(image,save_path):
    """
    画像内の各クラス領域に対して長さを測定するmain関数
    """
    length_list = []  # 長さリスト
    for num in range(1, np.max(image) + 1):
        # 値に対応する領域を抽出
        region = (image == num)
        if not np.any(region):
            continue
        #等高線の作成
        concat_array = find_contour_line(region)
        #長さの測定
        length = cal_cell_length(concat_array)
        length_list.append([num, length])
    #長さリストをヒストグラムで表し保存
    if length_list:
        values = [entry[1] for entry in length_list if entry[1] is not None]
    #     # ヒストグラムを作成
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(values, bins=np.arange(min(values), max(values) + 2) - 0.5, edgecolor='black')
    #     plt.xlim(right = 50)
    #     plt.ylim(top = 160)
    #     plt.xlabel("cell length")
    #     plt.ylabel("Count")
    #     plt.title('Histogram of cell length')
    #     plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show
        # plt.savefig(save_path + "/cal_cell_histogram2.png")
    return length_list, values

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
    padded_array = np.zeros(padded_shape, dtype=np.uint8)
    region_offset = coords - min_coords + 1
    for coord in region_offset:
        padded_array[tuple(coord)] = 1
    #抽出した領域を境界線から中心に向かってランク付け
    num=1
    back_img = np.zeros_like(padded_array)
    padded_array = copy.deepcopy(padded_array)
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
    return back_img


def cal_cell_length(image):
    """
    グラフの作成、長さの測定、長さの採用の判定のmain関数
    """
    graph = nx.Graph()
    graph = make_graph(image,graph)
    start_end_points = find_start_and_end_nodes(image, graph)  
    shortpath_list = []
    for i in range(len(start_end_points)):
        for j in range(i + 1, len(start_end_points)):  
            short_path = find_path(graph,start_end_points[i], start_end_points[j])
            if short_path == None:
                continue
            shortpath_list.append([len(short_path),start_end_points[i],start_end_points[j]])
    try:
        if shortpath_list:
            return np.max(np.array(shortpath_list)[:,0])
        else:
            print('error')
            return 1
    except:
        print('None')

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
                    graph.add_node(node_id, pos=(y, x), weight=1/image[y, x])  # ノード追加と重み設定
                    node_id += 1

                current_node = pixel_to_node[(y, x)]

                # 8近傍の探索
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and image[ny, nx] > 0:
                        if (ny, nx) not in pixel_to_node:
                            pixel_to_node[(ny, nx)] = node_id
                            graph.add_node(node_id, pos=(ny, nx), weight=1/image[ny, nx])  # ノード追加と重み設定
                            node_id += 1

                        neighbor_node = pixel_to_node[(ny, nx)]
                        if not graph.has_edge(current_node,neighbor_node):
                            edge_weight = ((1/image[y, x]) + (1/image[ny, nx]))  # エッジの重みをノード値で設定
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
        return None

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

def count_lengh_cal(image_path,save_path):
    np.random.seed(3407)
    image = load_image(image_path)
    # 実行
    result_image, extracted_regions = process_main(image)
    result_image = compress_array(result_image)
    # save_change_color_image(save_path, result_image)
    print(f'result_image unique: {np.unique(result_image)}')
    np.savetxt(save_path + '/gmmmml_np_savetxt3.csv', result_image, delimiter=',', fmt='%d')
    print(f'max: {np.max(result_image)}')
    length_list, values = cal_lengh_main(result_image,save_path)
    print(len(length_list))
    # CSVファイルに保存
    save_to_csv(length_list, save_path + "/length_list_gmmmml3.csv", headers=["class_number","Row Count"])

# class count_cell
def count_cell(image_path, save_path):
    np.random.seed(3407)
    image = load_image(image_path)
    # 実行
    result_image, extracted_regions = process_main(image)
    result_image = compress_array(result_image)
    # save_change_color_image(save_path, result_image)
    print(f'result_image unique: {np.unique(result_image)}')
    np.savetxt(save_path + '/gmmmml_np_savetxt3.csv', result_image, delimiter=',', fmt='%d')
    with open(save_path + '/gmmmml_cell_num.txt', 'w') as f:
        f.write(f'{np.max(result_image)}\n')
    print(f'max: {np.max(result_image)}')

def main():  
    pic =  Picture()
    np.set_printoptions(linewidth=200)
    #全体画像
    input_dir = '/home/nakajima/work/Ecoli/data/nd049_label_image/'
    image_path = input_dir + '8bit_nd049_S1_C1_T0.tif' # 8bit Grayscale画像
    # save_path = '/home/nakajima/work/Ecoli/code/BaSCA/predict/8bit_nd049_S1_C1_T0_test4.tif'
    mask_path = input_dir + 'manual_label/image/' + '8bit_nd049_S1_C1_T0.png' # 生細胞正解ラベル画像
    linemask_path = input_dir + 'line_label2/'+ '8bit_nd049_S1_C1_T0_binari.tiff'  #lineの正解ラベル画像
    predict_path = '/home/nakajima/work/Ecoli/code/Unet/result/20241212_205200_64_16_con/combined_image_or.png'


    # path = '/home/nakajima/work/Ecoli/code/Unet/data/64_16/test_manual/label/'
    # input_path = path + '00059.tif'#分割の正解ラベル画像
    path = '/home/nakajima/work/Ecoli/code/Unet/result/20241212_205200_64_16_con/image'
    # n, image_list = pic.get_picture(path)
    # basca_main(input_path,save_path)
    # for i,image_name in enumerate(image_list):
    #     print(i)
    #     test2(image_name,save_path,i)
    # save_path = '/home/nakajima/work/Ecoli/code/BaSCA/20241212_205200_64_16_con_2'
    # num = 4 
    # os.makedirs('/home/nakajima/work/Ecoli/code/BaSCA/20241212_205200_64_16_con_2/crop_classimage4',exist_ok=True)
    # test2(predict_path, save_path,0)
    # save_path = '/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250608_181159'
    save_path = '/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2'
    # p = os.path.join(save_path , 'large_label.png')
    p = os.path.join(save_path , '8bit_nd049_S1_C1_T0_label.tif')
    count_lengh_cal(p, save_path)
    # test_basca_main(input_path,save_path)
    # norm_image = cv2.applyColorMap(norm_image,cv2.COLORMAP_JET)
    # cv2.imwrite('/home/nakajima/work/Ecoli/code/seg_edge_length/prop_edge_length_image/findcontours.png',norm_image)

if __name__ == "__main__":
    main()


