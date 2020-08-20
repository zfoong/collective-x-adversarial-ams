# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:56:47 2020

@author: zfoong
"""

# Code used to generate FIG. 7 in thesis. show progression of active work of 
# all agents flocking agents only and clustering agents only

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data'
result_file_name_list = ['result_all.csv','result_c.csv','result_f.csv']
dir_list = [ 'data_1597631416']

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def moving_average(a, n=13) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    std = []
    for x in range(len(a[0])-n+1):
        std.append(np.std(a[0][x:x+n]))
    return (ret[n - 1:] / n, std)

if (__name__ == '__main__'):  
    ax = plt.subplot()
    col = ['blue', 'goldenrod', 'red']
    c = 0
    for path in dir_list:
        for result_file_name in result_file_name_list:
            new_path = os.path.join(data_path, path)
            sub_path_list = listdirs(new_path)
            result_list_dict = []
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
                    result_list = np.array(result_list)
                result_list_dict.append(result_list)
            result_list_dict = np.array(result_list_dict)
            avg_result = np.mean(result_list_dict, axis=0)
            avg_result, avg_std = moving_average(avg_result)
            print('avg of last 10 episode of ' + path)
            print(np.mean(avg_result[-10:]))
            liasta = range(0, len(avg_result))
            plt.errorbar(x=liasta,y=avg_result, yerr=avg_std, fmt='-', color=col[c], elinewidth = 0.5)
            c = c+1
    plt.ylabel("Active Work $aw(τ)$")
    plt.xlim(0,107)
    plt.xlabel("Episodes")
    plt.show()
    
    
    