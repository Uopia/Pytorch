from tkinter import Frame
import cv2
import numpy as np
import torch
import time
import pandas as pd
import os
import copy


class ModelEvaluation:

    def __init__(self, folder, model_name, results, img_paths, cf, use_name):

        self.labels = {'AJ': np.zeros([3, 2]), 'BX': np.zeros([3, 2]), 'CJ': np.zeros([3, 2]),
                       'CK': np.zeros([3, 2]), 'CQ': np.zeros([3, 2]), 'CR': np.zeros([3, 2]),
                       'FS': np.zeros([3, 2]), 'FZ': np.zeros([3, 2]), 'JG_D': np.zeros([3, 2]),
                       'PL_P': np.zeros([3, 2]), 'QF': np.zeros([3, 2]), 'SG': np.zeros([3, 2]),
                       'SL': np.zeros([3, 2]), 'TL': np.zeros([3, 2]), 'ZW': np.zeros([3, 2]),
                       'JG_U': np.zeros([3, 2]), 'PL_L': np.zeros([3, 2])}
        self.labels_list = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'JG_D', 'PL_P', 'QF', 'SG',
                            'SL', 'TL', 'ZW', 'JG_U', 'PL_L']
        self.folder = folder
        self.model_name = model_name
        self.use_name = set(use_name)
        self.unuse_name = set(self.labels_list).difference(self.use_name)
        self.results = results
        self.img_paths = img_paths
        self.len = len(self.labels_list)
        self.labels_cf = np.zeros([1, self.len])[0]+cf
        # self.images_path = os.path.join(folder, img_folder)
        self.labels_path = os.path.join(folder, 'labels')
        # self.names_suf = os.listdir(self.images_path)
        self.sum = np.zeros([3, 2])
        self.average_weight = [1, 1]

    def calculate(self):

        results_set = set()
        labels_set = set()
        self.labels = {'AJ': np.zeros([3, 2]), 'BX': np.zeros([3, 2]), 'CJ': np.zeros([3, 2]),
                       'CK': np.zeros([3, 2]), 'CQ': np.zeros([3, 2]), 'CR': np.zeros([3, 2]),
                       'FS': np.zeros([3, 2]), 'FZ': np.zeros([3, 2]), 'JG_D': np.zeros([3, 2]),
                       'PL_P': np.zeros([3, 2]), 'QF': np.zeros([3, 2]), 'SG': np.zeros([3, 2]),
                       'SL': np.zeros([3, 2]), 'TL': np.zeros([3, 2]), 'ZW': np.zeros([3, 2]),
                       'JG_U': np.zeros([3, 2]), 'PL_L': np.zeros([3, 2])}
        res_coos = []
        lab_coos = []

        for img_path, img_result in zip(self.img_paths, self.results):
            result = img_result[0]
            name_suf = img_path.split("\\")[-1]
            img_cvt = cv2.imread(img_path)

            for name_num in np.arange(self.len):
                res_lists = result[name_num]
                if res_lists.size > 0:
                    for res_list in res_lists:
                        if res_list[-1] > self.labels_cf[name_num]:
                            results_set.add(self.labels_list[name_num])
                            x = np.hstack((res_list[0:4], np.array(self.labels_list[name_num], dtype=object)))
                            res_coos.append(x)

            names = os.path.splitext(name_suf)[0]
            label_path = os.path.join(self.labels_path, names+'.txt')

            val = []
            if os.path.isfile(label_path):
                f = open(label_path, "r", encoding="UTF-8")
                val = f.readlines()
                f.close()

            for line in val:
                label = self.labels_list[int(line.split(" ")[0])]
                nums = []

                for num in line.split(" ")[1:5]:
                    nums.append(float(num))

                nums.append(label)
                lab_coos.append(nums)
                labels_set.add(label)

            r_l = results_set.intersection(labels_set)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if not results_set.symmetric_difference(labels_set):

                img_cvt_c = copy.deepcopy(img_cvt)
                w = img_cvt_c.shape[1]
                h = img_cvt_c.shape[0]
                for res_coo in res_coos:
                    img_cvt = cv2.rectangle(img_cvt, (int(min(res_coo[0], res_coo[2])),
                                                      int(min(res_coo[1], res_coo[3]))),
                                  (int(max(res_coo[0],res_coo[2])), int(max(res_coo[1],res_coo[3]))),
                                  color=(0, 0, 255), thickness=5)
                    img_cvt = cv2.putText(img_cvt, res_coo[-1], (int(min(res_coo[0], res_coo[2]))+10,
                                                                 int(max(res_coo[1], res_coo[3]))-10),
                                          font, 0.8, (0, 255, 255), 3)

                for lab_coo in lab_coos:

                    img_cvt_c = cv2.rectangle(img_cvt_c, (int(w*(lab_coo[0]-lab_coo[2]/2)),
                                                          int(h*(lab_coo[1]-lab_coo[3]/2))),
                                  (int(w*(lab_coo[0]+lab_coo[2]/2)), int(h*(lab_coo[1]+lab_coo[3]/2))),
                                  color=(255, 0, 0), thickness=5)
                    img_cvt_c = cv2.putText(img_cvt_c, lab_coo[-1], (int(w*(lab_coo[0]-lab_coo[2]/2))+10,
                                                                     int(h*(lab_coo[1]+lab_coo[3]/2))-10),
                                          font, 0.8, (255, 255, 0), 3)

                mage = np.zeros((h, w*2, 3), np.uint8)
                mage[0:h, 0:w] = img_cvt
                mage[0:h, w:2*w] = img_cvt_c
                cv2.imwrite(f'{self.folder}/page_g/{names}.jpg', mage)

            if not r_l:

                img_cvt_c = copy.deepcopy(img_cvt)
                w = img_cvt_c.shape[1]
                h = img_cvt_c.shape[0]
                for res_coo in res_coos:
                    img_cvt = cv2.rectangle(img_cvt, (int(min(res_coo[0], res_coo[2])),
                                                      int(min(res_coo[1], res_coo[3]))),
                                            (int(max(res_coo[0], res_coo[2])), int(max(res_coo[1], res_coo[3]))),
                                            color=(0, 0, 255), thickness=5)
                    img_cvt = cv2.putText(img_cvt, res_coo[-1], (int(min(res_coo[0], res_coo[2])) + 10,
                                                                 int(max(res_coo[1], res_coo[3])) - 10),
                                          font, 0.8, (0, 255, 255), 3)

                for lab_coo in lab_coos:
                    img_cvt_c = cv2.rectangle(img_cvt_c, (int(w * (lab_coo[0] - lab_coo[2] / 2)),
                                                          int(h * (lab_coo[1] - lab_coo[3] / 2))),
                                              (int(w * (lab_coo[0] + lab_coo[2] / 2)),
                                               int(h * (lab_coo[1] + lab_coo[3] / 2))),
                                              color=(255, 0, 0), thickness=5)
                    img_cvt_c = cv2.putText(img_cvt_c, lab_coo[-1], (int(w * (lab_coo[0] - lab_coo[2] / 2)) + 10,
                                                                     int(h * (lab_coo[1] + lab_coo[3] / 2)) - 10),
                                            font, 0.8, (255, 255, 0), 3)

                mage = np.zeros((h, w * 2, 3), np.uint8)
                mage[0:h, 0:w] = img_cvt
                mage[0:h, w:2 * w] = img_cvt_c
                cv2.imwrite(f'{self.folder}/page_b/{names}.jpg', mage)

            for lab in r_l:
                self.labels[lab][0][0] += 1
                if lab not in self.unuse_name:
                    self.sum[0][0] += 1

            for lab in results_set:
                self.labels[lab][0][1] += 1
                if lab not in self.unuse_name:
                    self.sum[0][1] += 1

            for lab in labels_set:
                self.labels[lab][1][0] += 1
                if lab not in self.unuse_name:
                    self.sum[1][0] += 1

            if self.sum[1][0] != 0:
                self.sum[2][0] = self.sum[0][0] / self.sum[1][0]
            if self.sum[0][1] != 0:
                self.sum[2][1] = self.sum[0][0] / self.sum[0][1]

            results_set.clear()
            labels_set.clear()
            lab_coos.clear()
            res_coos.clear()

            print("计算进度:", (self.img_paths.index(img_path)+1)/len(self.img_paths))
            print("sum:", self.sum)

        for name in self.labels_list:

            if self.labels[name][1][0] != 0:
                self.labels[name][2][0] = float(self.labels[name][0][0]/self.labels[name][1][0])

            if self.labels[name][0][1] != 0:
                self.labels[name][2][1] = float(self.labels[name][0][0] / self.labels[name][0][1])

        self.sum[2][0] = self.sum[0][0]/self.sum[1][0]
        self.sum[2][1] = self.sum[0][0] / self.sum[0][1]

    def print_res(self):

        for name in self.use_name:
            print(f"{name}:", self.labels[name])

        print("sum:", self.sum)

    def output_log(self):

        i = 1
        while os.path.isfile(f"{self.folder}/output_log/{self.model_name}_log_{i}.txt"):
            i += 1

        f = open(f"{self.folder}/output_log/{self.model_name}_log_{i}.txt", "a", encoding="UTF-8")

        f.write(f"{self.model_name}:\n")
        f.write(f"ave_w:\n{self.average_weight}:\n")
        f.write(f"confidence:\n{self.labels_cf}:\n")
        for name in self.use_name:
            f.write(f"{name}:\n {self.labels[name]}\n")
        f.write(f"sum:\n{self.sum}\n")
        f.close()



# l1.calculate()
# l1.print_res(['AJ', 'BX', 'CJ', 'CK', 'CQ', 'FS', 'FZ', 'JG_Down', 'QF', 'SG',
#               'SL', 'TL', 'ZW'])
# l1.output_log(['AJ', 'BX', 'CJ', 'CK', 'CQ', 'FS', 'FZ', 'JG_Down', 'QF', 'SG',
#               'SL', 'TL', 'ZW'])


