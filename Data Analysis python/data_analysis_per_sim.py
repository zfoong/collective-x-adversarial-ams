# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:56:47 2020

@author: zfoong
"""

# code used to generated FIG. 4, 5, 6. shows active work averaged over
# all simulation thru out 120 episodes

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

def moving_average(a, n=13) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    std = []
    for x in range(len(a[0])-n+1):
        std.append(np.std(a[0][x:x+n]))
    return (ret[n - 1:] / n, std)

if (__name__ == '__main__'):  
    ax = plt.subplot()
    for path in dir_list:
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
        #plt.plot(avg_result)
        print('avg of last 10 episode of ' + path)
        print(np.mean(avg_result[-10:]))
        liasta = range(0, len(avg_result))
        plt.errorbar(x=liasta,y=avg_result, yerr=avg_std, fmt='-', color='goldenrod', elinewidth = 0.5)
    plt.ylabel("Active Work $aw(τ)$")
    plt.xlim(0,107)
    plt.ylim(0,1)
    plt.xlabel("Episodes")
    plt.show()
    
    
    