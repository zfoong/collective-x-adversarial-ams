# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:37:43 2020

@author: zfoong
"""

# code used to generated FIG. 6, 7. Shows heat map of the progression of state
# value thru out 120 episodes

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data\data_1597327898\sim_6'
result_file_name = 'SVTable_c.csv'
radiansPiece =  np.pi * 2 / 20;

def listdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def indexToAction(index):
    return (radiansPiece * index) - np.pi;

if (__name__ == '__main__'):  
    ep_path = listdirs(data_path)
    ep_path = sorted(ep_path, key=lambda x: int(x[x.rfind('_')+1:]))
    state_matrix = []
    for path in ep_path:
        path = os.path.join(data_path, path)
        file = os.path.join(path, result_file_name)
        result = []
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                result.append(float(row[0]))
            result = np.array(result)
        state_matrix.append(result)
    state_matrix = np.transpose(state_matrix)
    state_matrix = (state_matrix - state_matrix.min(0)) / state_matrix.ptp(0)
    state_matrix = np.flip(state_matrix, 0)
    fig, ax = plt.subplots()
    plt.imshow(state_matrix, aspect='auto')
    plt.ylabel("State")
    plt.xlabel("Episodes")
    plt.yticks([0,5,10,15])
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = '$\pi$'
    labels[1] = '$\pi/2$'
    labels[2] = '0'
    labels[3] = '$-\pi/2$'
    ax.set_yticklabels(labels)
    plt.show()