# 将图片文件夹中的图片进行检测，并将结果输出到指定文件夹中（）
from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from Model_evaluation_cl_TF_231104 import ModelEvaluation
import numpy as np
import shutil
import torch
from tqdm import tqdm

config = r'D:\Desktop\Swin\Swin-Transformer-Object-Detection\work_dirs\mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint = r'D:\Desktop\Swin\Swin-Transformer-Object-Detection\work_dirs\epoch_26.pth'
img_folder = r'D:\Desktop\PIC_VIDEO\V2P\6'
# img_folder = r'D:\Desktop\Swin\Data\val\images'
device = 'cuda:0'
s_folder = r'D:\Desktop\PIC_VIDEO\V2P\6'
# use_label = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'PL_P', 'QF', 'SG', 'SL', 'TL', 'ZW', 'JG_U', 'PL_L']
# use_label = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'JG_D', 'PL_P', 'QF', 'SG', 'SL', 'TL', 'ZW', 'JG_U', 'PL_L']
use_label = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'SG', 'SL', 'TL', 'ZW', 'JG_U']
cf = 0.3

def main():
    score_thr = 0.1
    results = []

    img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    model = init_detector(config, checkpoint, device=device)

    for img_path in tqdm(img_paths):
        try:
            results.extend(inference_detector(model, img_path))
            
        except Exception as e:
            print(f"Error detected in image: {img_path}")
            print(f"Error message: {e}")
        
        torch.cuda.empty_cache()  # 清理CUDA内存
    a = results[::2]
    img_local = img_paths

    m = ModelEvaluation(s_folder, "COCO", a, img_local, cf, use_label)
    m.calculate()
    # m.print_res()
    # m.output_log()
    # show_result_pyplot(model, img_paths, img_result, score_thr=score_thr)

if __name__ == '__main__':
    main()
