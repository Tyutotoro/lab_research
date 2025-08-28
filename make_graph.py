import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import os
from PIL import Image
from scipy.stats import gaussian_kde
import re


def make_length_graph(pre_path, pre_name,scale_up = False):
    # CSVファイルを読み込む
    # true_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/line_label3/8bit_nd049_S1_C1_T0_length.csv"
    # true_data = pd.read_csv(true_path)
    # true_label_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/length_list.csv" #古い方  
    # true_label_path ="/home/nakajima/work/Ecoli/code/BaSCA/manual_label_ver3/length_list_gmmmml3.csv"
    # true_label_path ="/home/nakajima/work/Ecoli/code/BaSCA/manual_scalelabel_ver2/length_list_gmmmml3.csv"
    true_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250508_003126_256_nopre_con/length_list_gmmmml3.csv"
    true_label_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250713_164357/test/length_list_gmmmml3.csv"

    true_data = pd.read_csv(true_path)
    label_data = pd.read_csv(true_label_path) 
    pred_data = pd.read_csv(pre_path + pre_name)
    

    # NaNを除去した列を取得
    # true_count = true_data['length'].dropna().astype(int)
    true_count = true_data['Row Count'].dropna()
    label_count = label_data['Row Count'].dropna()
    pred_count = pred_data['Row Count'].dropna()
    # true_count = (true_count*3).astype(int)

    if scale_up == True:
        true_count = (true_count/3).astype(int)
        label_count = (label_count/3).astype(int)
        pred_count = (pred_count/3).astype(int)

    # 各データの出現頻度を計算
    unique_true = true_count.value_counts().sort_index()
    unique_label = label_count.value_counts().sort_index()
    unique_pred = pred_count.value_counts().sort_index()
    unique_true_numpy = unique_true.to_numpy()
    unique_label_numpy = unique_label.to_numpy()
    unique_pred_numpy = unique_pred.to_numpy()
    unique_true_dict = dict(zip(unique_true.index.values,unique_true_numpy))
    unique_label_dict =dict(zip(unique_label.index.values,unique_label_numpy))
    unique_pred_dict = dict(zip(unique_pred.index.values,unique_pred_numpy))
    
    diff_score = calc_diff_distri(unique_true_dict, unique_pred_dict)
    diff_score2 = calc_diff_distri(unique_true_dict, unique_label_dict)
    print(diff_score)
    print(diff_score2)
    # print(calc_js_div(unique_true_numpy, unique_pred_numpy))

    # unique_true_indexes = unique_true.index
    # unique_label_indexes = unique_label.index
    # unique_pred_indexes = unique_pred.index

    series_list = [unique_true, unique_label, unique_pred]
    # 全Seriesのインデックスから最大値を求める
    all_indexes = pd.Index([])
    for s in series_list:
        all_indexes = all_indexes.union(s.index)

    max_index = all_indexes.max()
    # 補完されたSeriesを格納するリスト
    filled_series_list = []

    # 各Seriesを補完
    for s in series_list:
        full_index = range(0, max_index + 1)
        s_filled = s.reindex(full_index, fill_value=0)
        filled_series_list.append(s_filled)
    length_true_num  = len(true_count)
    length_label_num = len(label_count)
    length_pred_num = len(pred_count)

    name_list = [
        f'delta acrA (n={length_true_num })',
        f'delta acrB (n={length_label_num})',
        f'delta acrR (n={length_pred_num})',
    ]
    # グラフ描画
    plt.figure(figsize=(20, 12))
    plt.plot(filled_series_list[0].index, filled_series_list[0].values, color='black', label=name_list[0])
    plt.plot(filled_series_list[1].index, filled_series_list[1].values, linestyle='--',color='black', label=name_list[1])
    plt.plot(filled_series_list[2].index, filled_series_list[2].values, color='#005AFF',  label=name_list[2])
    # plt.plot(unique_label.index, unique_label.values, linestyle='--',color='black', label=f'weight True Label (n={length_label_num})')
    # plt.plot(unique_pred.index, unique_pred.values, color='#005AFF',  label=f'no weight True Label (n={length_pred_num})')
    # plt.plot(filled_series_list[0].index, filled_series_list[0].values, color='black', label=f'True Label (n={length_true_num })')
    # plt.plot(filled_series_list[2].index, filled_series_list[2].values, linestyle='--',color='black', label=f'thinning method (n={length_label_num})')
    # plt.plot(filled_series_list[1].index, filled_series_list[1].values, color='#005AFF',  label=f'proposal method (n={length_pred_num})')
    # plt.scatter(filled_series_list[0].index, filled_series_list[0].values, color='black', label=f'True Label (n={length_true_num })')
    # plt.scatter(filled_series_list[1].index, filled_series_list[1].values, color='red', label=f'weight True Label (n={length_label_num})')
    # plt.scatter(filled_series_list[2].index, filled_series_list[2].values, color='#005AFF',  label=f'no weight True Label (n={length_pred_num})')
    
    # plt.plot(unique_pred.index, unique_pred.values, color='#005AFF',  label=f'new method manual Label (n={length_pred_num})')

    x_max = max(unique_true.index.max(), unique_label.index.max(), unique_pred.index.max())
    # グラフの装飾
    plt.title("")
    plt.xlabel("length (pixels)")
    plt.ylabel("population (pieces)")
    plt.ylim(0,350)
    # plt.ylim(0,200)
    # plt.xticks(range(0, max_index + 1, 1))
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.legend(loc='upper right',fontsize=28)
    # plt.text((x_max-10), 200, f'Histogram Absolute Error:{diff_score2}', va='center', fontsize= 14,fontweight="bold", backgroundcolor="lightgray")
    # plt.text((x_max-10), 125, f'no weight HAE:{diff_score}', va='center', fontsize= 14,fontweight="bold", backgroundcolor="lightgray")
    # plt.text((x_max-10), 100, f'weight HAE:{diff_score2}', va='center', fontsize= 14,fontweight="bold", backgroundcolor="lightgray")
    # plt.text((x_max-40), 150, f'Histogram Absolute Error:{diff_score}', va='center', fontsize= 24,fontweight="bold", backgroundcolor="lightgray")

    # 保存と表示
    save_path = os.path.join(pre_path, 'length_graph_KO_3bai.png')
    print(save_path)
    plt.savefig(save_path)
    plt.show()


def make_boxplot(save_path):
    # acrA_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250508_003126_256_nopre_con/length_list_gmmmml3.csv"
    acrA_path_1 = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250715_163717_manual2/test/length_list_gmmmml3.csv"
    acrB_path_1 = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250713_164357_acrB_T1/test/length_list_gmmmml3.csv"
    acrR_path_1 = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250713_164701_acrR_T1/test/length_list_gmmmml3.csv"
    wt_path_1 = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250722_144023_WT_T1/test/length_list_gmmmml3.csv"

    base_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/"
    acrA_path = base_path + "Ecoli_20250715_171539_acrA_T30/test/length_list_gmmmml3.csv"
    acrB_path = base_path + "Ecoli_20250715_160750_acrB_T30/test/length_list_gmmmml3.csv"
    acrR_path = base_path + "Ecoli_20250715_160619_acrR_T30/test/length_list_gmmmml3.csv"
    wt_path = base_path + "Ecoli_20250715_161016_WT_T30/test/length_list_gmmmml3.csv"

    acrA_data_1 = pd.read_csv(acrA_path_1)
    acrB_data_1 = pd.read_csv(acrB_path_1) 
    acrR_data_1 = pd.read_csv(acrR_path_1)
    wt_data_1 = pd.read_csv(wt_path_1)
    acrA_data = pd.read_csv(acrA_path)
    acrB_data = pd.read_csv(acrB_path) 
    acrR_data = pd.read_csv(acrR_path)
    wt_data = pd.read_csv(wt_path)

    # NaNを除去した列を取得
    # true_count = true_data['length'].dropna().astype(int)
    acrA_count_1 = acrA_data_1['Row Count'].dropna()
    acrB_count_1 = acrB_data_1['Row Count'].dropna()
    acrR_count_1 = acrR_data_1['Row Count'].dropna()
    wt_count_1 = wt_data_1['Row Count'].dropna()
    acrA_count = acrA_data['Row Count'].dropna()
    acrB_count = acrB_data['Row Count'].dropna()
    acrR_count = acrR_data['Row Count'].dropna()
    wt_count = wt_data['Row Count'].dropna()
    all_data = [wt_count_1, wt_count,
                acrA_count_1, acrA_count,
                acrB_count_1, acrB_count,
                acrR_count_1, acrR_count]
    
    all_data = [(i/3) * 0.28 for i in all_data]
    lengths = [len(d) for d in all_data]

    fig, ax = plt.subplots(figsize = (23,15))
    positions = []
    for i in range(len(all_data)):
        positions.append(i*0.6)
    box = ax.boxplot(all_data,
                    positions=positions,
                    patch_artist=True,
                    widths=0.4,  # boxの幅の設定
                    medianprops=dict(color='orange', linewidth=3),  # 中央値の線の設定
                    whiskerprops=dict(color='black', linewidth=2),  # ヒゲの線の設定
                    capprops=dict(color='black', linewidth=1),  # ヒゲの先端の線の設定
                    sym = '',
                    # showmeans=True, meanline=True,
                    # meanprops=dict(color = 'green', linewidth = 3)
                    )
    colors=['#8A2BE280']

    # boxの色の設定
    for i, b in enumerate(box['boxes']):
        if i % 2 == 1:
            b.set(color='black', linewidth=1)  # boxの外枠の色
            b.set_facecolor('gray') # boxの色
        else:
            b.set(color='black', linewidth=1)  # boxの外枠の色
            b.set_facecolor('white') # boxの色

    group_positions = [(positions[0] + positions[1]) / 2,
                    (positions[2] + positions[3]) / 2,
                    (positions[4] + positions[5]) / 2,
                    (positions[6] + positions[7]) / 2,
                ]
    labels = ['WT', 'acrA', 'acrB', 'acrR']
    ax.set_ylim(0,4.5)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels, fontsize=30)

    # 第1四分位数と第3四分位数の取得（最初の箱に注目）
    q1 = box['boxes'][0].get_path().vertices[1][1]  # bottom of box = Q1
    q3 = box['boxes'][0].get_path().vertices[2][1]  # top of box = Q3


    # 横点線を描画（x 軸の範囲を広げて横線にする）
    x_min, x_max = ax.get_xlim()
    ax.hlines([q1, q3], xmin=x_min, xmax=x_max, colors='gray', linestyles='dotted', label='Q1/Q3 lines')

    plt.show()

    fig, ax = plt.subplots(figsize = (23,14))
    positions = [1,2,3,4,5,6,7,8]
    group_positions = [(positions[0] + positions[1]) / 2,
                        (positions[2] + positions[3]) / 2,
                        (positions[4] + positions[5]) / 2,
                        (positions[6] + positions[7]) / 2,
                    ]
    # labels = ['WT', 'acrA', 'acrB', 'acrR']

    color_list = ["gray" if i % 2 == 1 else "white" for i in range(len(labels))]
    plt.bar(positions,lengths,color = color_list,edgecolor='black')
    ax.tick_params(axis='y', labelsize=25)
    plt.xticks(group_positions, labels,fontsize = 30)
    plt.show()
    
#細胞帳のヒストグラム生成
def make_histgram(pre_path,pre_name):
    # CSVファイルを読み込む
    label_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/line_label3/8bit_nd049_S1_C1_T0_length.csv"
    data_label = pd.read_csv(label_path)

    file_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/length_list.csv"  
    data = pd.read_csv(file_path) 

    data2 =pd.read_csv(pre_path + pre_name)

    # NaNを除去
    length_data_label = data_label['length'].dropna()
    length_data1 = data['Row Count'].dropna()
    length_data2 = data2['Row Count'].dropna()
    length_data2 = length_data2 / 3

    length_data_label_num = len(length_data_label.dropna())
    length_data1_num = len(length_data1.dropna())
    length_data2_num = len(length_data2.dropna())
    bins = np.arange(min(length_data2.min(), length_data_label.min(), length_data1.min()), 
                    max(length_data2.max(), length_data_label.max(), length_data1.max()) + 1, 1)  # 1ごとの区切り
    # ヒストグラムをプロット
    plt.figure(figsize=(10, 6))
    counts_label, _, _ = plt.hist(length_data_label, bins=bins, color='red', edgecolor='black', alpha=0.7, label=f'True Label (n={length_data_label_num})')
    counts_segmentation, _, _ = plt.hist(length_data1, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7, label=f'Segmentation Label (n={length_data1_num})')
    counts_proposal, _, _ = plt.hist(length_data2, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, label=f'Proposal Method (Scaled) (n={length_data2_num})')
    max_bin = bins.max()
    plt.xlim(right = max_bin)
    plt.xlabel('length (pixels)')
    max_count = max(counts_label.max(), counts_segmentation.max(), counts_proposal.max())
    plt.ylim(top=max_count * 1.1)
    plt.ylabel('population (pieces)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=20)
    # plt.savefig(os.path.join(pre_path + 'histgram.png'))
    plt.show()

def make_line_graph(base_dir_list,save_path):
    data_list = []
    pattern = re.compile(r'^T\d{4}$')
    for i, base_dir in enumerate(base_dir_list):
        data = []
        folder_names = []
        # exフォルダ内のサブフォルダを取得・フィルタ・ソート
        for name in os.listdir(base_dir):
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path) and pattern.match(name):
                folder_names.append(name)

        # TXXXX形式でソート（数値順）
        folder_names.sort(key=lambda x: int(x[1:]))

        # 各フォルダのlog.txtを読み込む
        for folder in folder_names:
            log_path = os.path.join(base_dir, folder, 'gmmmml_cell_num.txt')
            try:
                with open(log_path, 'r') as f:
                    value = float(f.read().strip())
                    data.append(value)
            except Exception as e:
                print(f'読み込み失敗: {log_path} ({e})')
                data.append(None)
        data_list.append(data)

    # x = range(len(data_list[0]))
    x = []
    i = 1
    t = 0
    while True:
        timepoint = 0
        if i <= 146:
            timepoint =  t+5
            t = timepoint
        elif i > 146:
            timepoint =  t+20
            t = timepoint
        # timepoint = timepoint/60
        x.append(timepoint)
        if i >= len(data_list[0]):
            break
        i += 1
    # グラフ描画
    plt.figure(figsize=(12, 6))
    plt.plot(x, data_list[0], color = 'black')
    plt.plot(x, data_list[1], color = '#005AFF')
    # plt.xlabel('time point')
    # plt.ylabel('cell num')
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cell_num_graph_WTacrR3.png'))
    plt.show()

def js_main(pre_path, pre_name,scale_up = False):
    true_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/line_label3/8bit_nd049_S1_C1_T0_length.csv"
    true_data = pd.read_csv(true_path)
    true_label_path = "/home/nakajima/work/Ecoli/data/nd049_label_image/manual_label2/length_list.csv"  
    label_data = pd.read_csv(true_label_path) 
    unet_label_path = '/home/nakajima/work/Ecoli/code/BaSCA/20241212_205200_64_16_con_2/length_list2.csv'
    unet_data = pd.read_csv(unet_label_path) 
    pred_data = pd.read_csv(pre_path + pre_name)

    # NaNを除去した列を取得
    true_count = true_data['length'].dropna().astype(int)
    label_count = label_data['Row Count'].dropna()
    unet_count = unet_data['Row Count'].dropna()
    pred_count = pred_data['Row Count'].dropna()
    if scale_up == True:
        pred_count = (pred_count/3).astype(int)

    # 各データの出現頻度を計算
    unique_true = true_count.value_counts().sort_index()
    unique_label = label_count.value_counts().sort_index()
    unique_unet = unet_count.value_counts().sort_index()
    unique_pred = pred_count.value_counts().sort_index()
    unique_true_numpy = unique_true.to_numpy()
    unique_label_numpy = unique_label.to_numpy()
    unique_unet_numpy = unique_unet.to_numpy()
    unique_pred_numpy = unique_pred.to_numpy()

    unique_true_dict = dict(zip(unique_true.index.values,unique_true_numpy))
    unique_label_dict = dict(zip(unique_label.index.values,unique_label_numpy))
    unique_unet_dict = dict(zip(unique_unet.index.values,unique_unet_numpy))
    unique_pred_dict = dict(zip(unique_pred.index.values,unique_pred_numpy))

    result = calc_diff_distri(unique_true_dict, unique_pred_dict)
    # print(result)

def calc_js_div(true_distri, pred_distri):
    """JSダイバージェンスを計算"""

    result1 = 0
    result2 = 0
    all_keys = set(true_distri.keys()).union(set(pred_distri.keys()))

    filled_true_distri = {key: true_distri.get(key, 0) for key in all_keys}
    filled_pred_distri = {key: pred_distri.get(key, 0) for key in all_keys}
    filled_true_distri_list = [x +1 for x in list(filled_true_distri.values())]
    filled_pred_distri_list = [x +1 for x in list(filled_pred_distri.values())]
    true_norm_list = filled_true_distri_list/np.sum(filled_true_distri_list)
    pred_norm_list = filled_pred_distri_list/np.sum(filled_pred_distri_list)
    print(true_norm_list)
    print(len(true_norm_list))
    print(pred_norm_list)
    print(len(pred_norm_list))
    for p, q in zip(true_norm_list, pred_norm_list):
        if q == 0:
            raise AssertionError("KL divergence can't handle zeros.")
        result1 += p * np.log(p / q)
    for p, q in zip(true_norm_list, pred_norm_list):
        if p == 0:
            raise AssertionError("KL divergence can't handle zeros.")
        result1 += q * np.log(q / p)

    return (result1/2) + (result2/2)

def calc_kl_div(true_distri, pred_distri):
    """KLダイバージェンスを計算"""
    result = 0
    true_list = []
    pred_list = []
    all_keys = set(true_distri.keys()).union(set(pred_distri.keys()))

    filled_true_distri = {key: true_distri.get(key, 0) for key in all_keys}
    filled_pred_distri = {key: pred_distri.get(key, 0) for key in all_keys}
    # true_list.extend(filled_true_distri.get(i) for i in pred_distri.keys())
    # pres_list.extend(filled_pred_distri.get(i) for i in )
    filled_true_distri_list = [x +1 for x in list(filled_true_distri.values())]
    filled_pred_distri_list = [x +1 for x in list(filled_pred_distri.values())]
    # print(filled_pred_distri, filled_true_distri)
    true_norm_list = filled_true_distri_list/np.sum(filled_true_distri_list)
    pred_norm_list = filled_pred_distri_list/np.sum(filled_pred_distri_list)
    print(true_norm_list)
    print(pred_norm_list)
    # pred_norm_list = list(filled_pred_distri.values())/np.sum(list(filled_pred_distri.values()))
    for p, q in zip(true_norm_list, pred_norm_list):
        if q == 0:
            raise AssertionError("KL divergence can't handle zeros.")
        result += p * np.log(p / q)
    return result

def calc_diff_distri(true_distri, pred_distri):
    diff_list = []
    all_keys = set(true_distri.keys()).union(set(pred_distri.keys()))

    filled_true_distri = {key: true_distri.get(key, 0) for key in all_keys}
    filled_pred_distri = {key: pred_distri.get(key, 0) for key in all_keys}
    filled_true_distri_list = [x  for x in list(filled_true_distri.values())]
    filled_pred_distri_list = [x  for x in list(filled_pred_distri.values())]
    for true_point, pred_point in zip(filled_true_distri_list, filled_pred_distri_list):
        diff_list.append(np.abs(pred_point-true_point))
    return np.sum(diff_list)


def main():    
    pre_path = "/home/nakajima/work/Ecoli/code/DUNet-retinal-vessel-detection/log/experiments/deform_unet_v1/Ecoli_20250713_164701/test/"
    # pre_path = '/home/nakajima/work/Ecoli/code/BaSCA/manual_scalelabel_ver2/'
    # pre_name = 'length_list_gmmmml_noweight.csv'
    pre_name = 'length_list_gmmmml3.csv'
    # make_length_graph(pre_path,pre_name,False)
    # make_length_graph(pre_path, pre_name, True)
    # make_boxplot(pre_path)
    base_dir_list = [
        '/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/log/experiments/deform_unet_v1/Ecoli_20250723_120549_WT_all',
        '/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/log/experiments/deform_unet_v1/Ecoli_20250725_224058_acrR_all'
    ]
    save_path = '/home/nakajima/work/Ecoli/code/DUNet_retinal_vessel_detection/log/experiments/deform_unet_v1/Ecoli_20250723_120549_WT_all'
    make_line_graph(base_dir_list,save_path)

if __name__ == "__main__":
    main()