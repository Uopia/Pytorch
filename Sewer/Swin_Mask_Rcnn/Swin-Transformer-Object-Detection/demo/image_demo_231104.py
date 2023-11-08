# 将图片文件夹中的图片进行检测，并将结果输出到指定文件夹中（），视频分割成图片并调用画图程序画框，再组合成视频
from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from Model_evaluation_cl_TF_231104 import ModelEvaluation
import numpy as np
import shutil
import torch
from tqdm import tqdm
import cv2
from moviepy.editor import VideoFileClip

config = r'D:\Desktop\mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint = r'D:\Desktop\epoch_26.pth'
v_path =r"D:\Desktop\11\demo.mp4"
device = 'cuda:0'
# use_label = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'PL_P', 'QF', 'SG', 'SL', 'TL', 'ZW', 'JG_U', 'PL_L']
# use_label = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'JG_D', 'PL_P', 'QF', 'SG', 'SL', 'TL', 'ZW', 'JG_U', 'PL_L']
use_label = ['AJ', 'BX', 'CJ', 'CK', 'CQ', 'CR', 'FS', 'FZ', 'SG', 'SL', 'TL', 'ZW', 'JG_U']
cf = 0.3


def v2p(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    name_suf = video_path.split("\\")[-1]
    video_folder = os.path.dirname(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        exit()

    # 创建一个文件夹来保存图像
    pout_folder = os.path.join(video_folder, f'{name_suf}_pout')
    os.makedirs(pout_folder, exist_ok=True)

    # 设置输出图像的帧速率
    output_frame_rate = 15  # 每秒15张图像
    output_interval = int(1 / output_frame_rate * 1000)  # 以毫秒为单位的时间间隔

    frame_count = 0
    output_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # 检查是否应该输出当前帧
        if frame_count % (cap.get(cv2.CAP_PROP_FPS) // output_frame_rate) == 0:
            output_count += 1
            output_filename = os.path.join(pout_folder, f"{name_suf}_{output_count:04d}.jpg")  # 图像文件名，保存在指定文件夹中
            cv2.imwrite(output_filename, frame)
            print(f"输出图像: {output_filename}")

        # 显示视频（可选）
        # cv2.imshow('Video', frame)

        # if cv2.waitKey(output_interval) & 0xFF == ord('q'):
        #     break

    # 释放资源
    cap.release()
    # cv2.destroyAllWindows()

    return pout_folder, video_folder


def p2v(video_path):

    name_suf = video_path.split("\\")[-1]
    video_folder = os.path.dirname(video_path)
    image_folder = os.path.join(video_folder, f'img_all')

    vout_folder = os.path.join(video_folder, f'{name_suf}_vout')  # 指定保存视频的文件夹名称

    # 创建保存视频的文件夹
    os.makedirs(vout_folder, exist_ok=True)

    # 获取图像文件列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 设置视频编解码器和帧速率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout_path = os.path.join(vout_folder, f"{name_suf}.mp4")  # 视频文件名，保存在指定文件夹中
    video = cv2.VideoWriter(vout_path, fourcc, 15, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # cv2.destroyAllWindows()
    video.release()

    print(f"已生成视频: {vout_path}")


def v2g(video_path):

    name_suf = video_path.split("\\")[-1]
    video_folder = os.path.dirname(video_path)
    vout_folder = os.path.join(video_folder, f'{name_suf}_vout')
    input_file = os.path.join(vout_folder, f"{name_suf}.mp4")
    output_file = os.path.join(vout_folder, f"{name_suf}.gif")

    fps = 10
    video_clip = VideoFileClip(input_file)
    gif_clip = video_clip.subclip(0, video_clip.duration)
    gif_clip = gif_clip.set_duration(video_clip.duration)
    gif_clip = gif_clip.set_fps(fps)
    gif_clip.write_gif(output_file)
    video_clip.close()


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


if __name__ == '__main__':

    img_folder, s_folder = v2p(v_path)
    main()
    p2v(v_path)
    v2g(v_path)