# 分割组合视频+分每类阈值
from tkinter import Frame
import cv2
import numpy as np
import torch
import time
import pandas as pd
import os
import copy
import csv

class ModelEvaluation:

    def __init__(self, folder, model_name, results, img_paths, cf, use_name):

        self.labels = {'AJ': np.zeros([3, 2]), 'BX': np.zeros([3, 2]), 'CJ': np.zeros([3, 2]),
                       'CK': np.zeros([3, 2]), 'CQ': np.zeros([3, 2]), 'CR': np.zeros([3, 2]),
                       'FS': np.zeros([3, 2]), 'FZ': np.zeros([3, 2]), 'JG_D': np.zeros([3, 2]),
                       'PL_P': np.zeros([3, 2]), 'QF': np.zeros([3, 2]), 'SG': np.zeros([3, 2]),
                       'SL': np.zeros([3, 2]), 'TL': np.zeros([3, 2]), 'ZW': np.zeros([3, 2]),
                       'JG_U': np.zeros([3, 2]), 'PL_L': np.zeros([3, 2])}
        self.labels_list = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'JG_D', 'PL_P', 'QF', 'SG', 'SL', 'TL', 'ZW', 'JG_U', 'PL_L']
        self.folder = folder
        self.model_name = model_name
        self.use_name = set(use_name)
        self.unuse_name = set(self.labels_list).difference(self.use_name)
        self.results = results
        # self.results = [result[0::2] for result in results]
        # self.results = results[::2]
        self.img_paths = img_paths
        self.len = len(self.labels_list)
        self.labels_cf = np.zeros([1, self.len])[0]+cf
        self.labels_cf[0] = 0.55
        self.labels_cf[1] = 0.6
        self.labels_cf[2] = 0.3
        self.labels_cf[3] = 0.45
        self.labels_cf[4] = 0.25
        self.labels_cf[5] = 0.5
        self.labels_cf[6] = 0.3
        self.labels_cf[7] = 0.05
        self.labels_cf[8] = 0.4
        self.labels_cf[9] = 0.3
        self.labels_cf[10] = 0.3
        self.labels_cf[11] = 0.7
        self.labels_cf[12] = 0.3
        self.labels_cf[13] = 0.5
        self.labels_cf[14] = 0.3
        self.labels_cf[15] = 0.4
        self.labels_cf[16] = 0.4
        # self.images_path = os.path.join(folder, img_folder)
        self.labels_path = os.path.join(folder, 'labels')
        # self.names_suf = os.listdir(self.images_path)
        self.sum = np.zeros([3, 2])
        self.average_weight = [1, 1]

    def calculate(self):

        results_set = set()
        labels_set = set()
        res_coos = []
        lab_coos = []

        for img_path, img_result in zip(self.img_paths, self.results):
            # result = img_result[0]
            result = img_result
            name_suf = img_path.split("\\")[-1]
            img_cvt = cv2.imread(img_path)

            for name_num in np.arange(self.len):
                res_lists = result[name_num]
                if res_lists.size > 0:
                    for res_list in res_lists:
                        if res_list[-1] > self.labels_cf[name_num]:
                            # print('name_num={}'.format(name_num))
                            results_set.add(self.labels_list[name_num])
                            x = np.hstack((res_list[0:4], np.array(self.labels_list[name_num], dtype=object)))
                            res_coos.append(x)


            names = os.path.splitext(name_suf)[0]


            font = cv2.FONT_HERSHEY_SIMPLEX

            img_cvt_c = copy.deepcopy(img_cvt)

            if not os.path.exists(f'{self.folder}/img_all'):
                os.makedirs(f'{self.folder}/img_all')
            
            for res_coo in res_coos:
                
                    img_cvt_c = cv2.rectangle(img_cvt_c, (int(min(res_coo[0], res_coo[2])),
                                                        int(min(res_coo[1], res_coo[3]))),
                                            (int(max(res_coo[0], res_coo[2])), int(max(res_coo[1], res_coo[3]))),
                                            color=(0, 0, 255), thickness=5)
                    img_cvt_c = cv2.putText(img_cvt_c, res_coo[-1], (int(min(res_coo[0], res_coo[2])) + 10,
                                                                    int(max(res_coo[1], res_coo[3])) - 10),
                                            font, 0.8, (0, 255, 255), 3)
                    
            cv2.imwrite(f'{self.folder}/img_all/{names}.jpg', img_cvt_c)

            results_set.clear()
            res_coos.clear()










