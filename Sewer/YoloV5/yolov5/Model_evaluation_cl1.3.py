# 该程序为评估程序，对.pth文件的性能进行评定。

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

    def __init__(self, model, folder, model_name, use_name, cf):
        # 加载模型
        self.model = model
        self.labels = {'AJ': np.zeros([3, 2]), 'BX': np.zeros([3, 2]), 'CJ': np.zeros([3, 2]),
                       'CK': np.zeros([3, 2]), 'CQ': np.zeros([3, 2]), 'CR': np.zeros([3, 2]),
                       'FS': np.zeros([3, 2]), 'FZ': np.zeros([3, 2]), 'JG_Down': np.zeros([3, 2]),
                       'PL': np.zeros([3, 2]), 'QF': np.zeros([3, 2]), 'SG': np.zeros([3, 2]),
                       'SL': np.zeros([3, 2]), 'TL': np.zeros([3, 2]), 'ZW': np.zeros([3, 2]),
                       'JG_Mid': np.zeros([3, 2]), 'JG_Up': np.zeros([3, 2]), 'PL_L': np.zeros([3, 2])}
        self.labels_list = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'JG_Down', 'PL', 'QF', 'SG',
                            'SL', 'TL', 'ZW', 'JG_Mid', 'JG_Up', 'PL_L']
        self.folder = folder
        self.model_name = model_name
        self.use_name = set(use_name)
        self.unuse_name = set(self.labels_list).difference(self.use_name)
        self.len = len(self.labels_list)
        self.labels_cf = np.zeros([1, self.len])[0]+cf
        self.images_path = os.path.join(folder, 'images')
        self.labels_path = os.path.join(folder, 'labels')
        self.names_suf = os.listdir(self.images_path)
        self.sum = np.zeros([3, 2])
        self.average_weight = [1, 1]

    def calculate(self):

        results_set = set()
        labels_set = set()
        self.labels = {'AJ': np.zeros([3, 2]), 'BX': np.zeros([3, 2]), 'CJ': np.zeros([3, 2]),
                       'CK': np.zeros([3, 2]), 'CQ': np.zeros([3, 2]), 'CR': np.zeros([3, 2]),
                       'FS': np.zeros([3, 2]), 'FZ': np.zeros([3, 2]), 'JG_Down': np.zeros([3, 2]),
                       'PL': np.zeros([3, 2]), 'QF': np.zeros([3, 2]), 'SG': np.zeros([3, 2]),
                       'SL': np.zeros([3, 2]), 'TL': np.zeros([3, 2]), 'ZW': np.zeros([3, 2]),
                       'JG_Mid': np.zeros([3, 2]), 'JG_Up': np.zeros([3, 2]), 'PL_L': np.zeros([3, 2])}
        res_coos = []
        lab_coos = []

        for name_suf in self.names_suf:

            # 这里是用导入的模型进行预测
            img_cvt = cv2.imread(os.path.join(self.images_path, name_suf))
            results = self.model(img_cvt)
            pd = results.pandas().xyxy[0]

            for name in self.labels_list:

                res_list = pd[pd['name'] == name].to_numpy()

                if res_list.size > 0:
                    for res_l in res_list:
                        if res_l[-3] > self.labels_cf[self.labels_list.index(name)]:
                            results_set.add(res_l[-1])
                            x = np.append(res_l[0:4], res_l[-1])
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

            print("计算进度:", (self.names_suf.index(name_suf)+1)/len(self.names_suf))
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
        for name in self.unuse_name:
            f.write(f"{name}:\n {self.labels[name]}\n")
        f.write(f"sum:\n{self.sum}\n")
        f.close()

    def save_to_csv(self):
        # 创建CSV文件名
        if self.folder == r'D:\Desktop':
            self.folder = os.path.join(self.folder, "evaluation_results_folder")
        csv_filename = os.path.join(self.folder, "evaluation_results.csv")

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


    # 创建CSV文件并写入数据
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # 写入类名
            label_names = list(self.labels.keys())
            csvwriter.writerow(['Class Names'] + label_names + ['Overall'])

            # 写入每个类的召回率
            recalls = [self.labels[name][2][0] for name in label_names]
            csvwriter.writerow(['Recall'] + recalls + [self.sum[2][0]])

            # 写入每个类的精确率
            precisions = [self.labels[name][2][1] for name in label_names]
            csvwriter.writerow(['Precision'] + precisions + [self.sum[2][1]])

        print(f"Results saved to {csv_filename}")


# 第一个参数是Yolo的模型地址，第三个参数是.pt文件的位置
# ModelEvauation（）的第二个参数是val的位置，val中的文件需要包括images、labels，子文件夹中没有下一级文件。 第四个参数是需要保留的标签名称，第五个参数是CF。
# 运行程序后会生成output_log、page_b和page_g。
# 同时生成一个.csv文件，文件是日志的表格模式。


model = torch.hub.load(r'D:\Desktop\YoloV5_T\SwinTransformer-YOLOv5-main', 'custom',
                       path=r'D:\Desktop\YoloV5\yolov51\yolov5\yolov5\yolo_831\base_l2\weights\best.pt', source='local')
# l1 = ModelEvaluation(model, r'D:\Desktop\YoloV5\val', 'yolov5_t', ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'JG_Down', 'PL', 'QF', 'SG',
#                                                                         'SL', 'TL', 'ZW', 'JG_Mid', 'JG_Up', 'PL_L'], 0.1)
l1 = ModelEvaluation(model, r'D:\Desktop\YoloV5\val', 'yolov5_831_l', ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'FS', 'FZ', 'JG_Down', 'QF', 'SG',
                                                                      'SL', 'TL', 'ZW'], 0.3)

l1.calculate()
l1.print_res()
l1.output_log()
l1.save_to_csv()


