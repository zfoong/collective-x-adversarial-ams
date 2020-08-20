# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:56:47 2020

@author: zfoong
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data'
 # 'result_f.csv' for flocking agents
 # 'result_c.csv' for clustering agents
result_file_name = 'result_all.csv'
dir_list = ['data_1597431145']

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if (__name__ == '__main__'):  
    for path in dir_list:
        new_path = os.path.join(data_path, path)
        sub_path_list = listdirs(new_path)
        for sub_path in sub_path_list:
            result_list = []
            sub_path = os.path.join(new_path, sub_path)
            file = os.path.join(sub_path, result_file_name)
            with open(file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                result = []
                for row in reader:
                    result.append(float(row[0]))
                result_list.append(result)
                plt.plot(result_list[0])
    plt.ylabel("Global Active Work w(s)")
    plt.xlabel("Episodes")
    plt.show()