import glob
import re
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import csv
import os


#クラス定義
class Search:
    #file読み込み
    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text) ]
    
    def sort_file(self,file):
        return sorted(file, key=self.natural_keys)
    
    def seq_file(self,file):
        source_file, target_file=[],[]
        files=(self.sort_file(file))
        num=len(files)
        for i in range(num):
            if '0_' in files[i]:
                source_file.append(files[i])
            elif 'src' in files[i]:
                source_file.append(files[i])
            else :
                target_file.append(files[i]) 
        return source_file, target_file
    
