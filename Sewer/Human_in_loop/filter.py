# 第二步，把文件夹中的文件按照类型筛选并移动到相应文件夹，文件夹中的文件名以a开头
import os
import shutil

def filter_and_organize_files(folder_path):
    # 创建子文件夹
    images_path = os.path.join(folder_path, 'images')
    labels_path = os.path.join(folder_path, 'labels')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        if filename.startswith('a'):
            file_path = os.path.join(folder_path, filename)
            # 检查文件类型并移动到相应文件夹
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(file_path, images_path)
            elif filename.lower().endswith('.txt'):
                shutil.move(file_path, labels_path)

# 设置文件夹路径
folder_path = 'D:\\Desktop\\lab_g\\ZW'
filter_and_organize_files(folder_path)
