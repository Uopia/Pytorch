#划分数据集
import os
import random
import shutil

# 定义源文件夹路径
data_folder = "D:/Desktop/Data"

# 定义训练集和验证集的比例
train_ratio = 0.8

# 创建目标文件夹
train_images_folder = os.path.join(data_folder, "train", "images")
val_images_folder = os.path.join(data_folder, "val", "images")
train_labels_folder = os.path.join(data_folder, "train", "labels")
val_labels_folder = os.path.join(data_folder, "val", "labels")

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# 获取所有图像文件的列表
images_folder = os.path.join(data_folder, "images")
image_files = os.listdir(images_folder)
random.shuffle(image_files)

# 根据比例分割训练集和验证集
split_index = int(len(image_files) * train_ratio)
train_images = image_files[:split_index]
val_images = image_files[split_index:]

# 将图像和标签文件复制到相应的文件夹
for image_file in train_images:
    # 复制图像文件
    src_image_path = os.path.join(images_folder, image_file)
    dst_image_path = os.path.join(train_images_folder, image_file)
    shutil.copy(src_image_path, dst_image_path)

    # 复制对应的标签文件
    label_file = os.path.splitext(image_file)[0] + ".txt"
    src_label_path = os.path.join(data_folder, "labels", label_file)
    dst_label_path = os.path.join(train_labels_folder, label_file)
    shutil.copy(src_label_path, dst_label_path)

for image_file in val_images:
    # 复制图像文件
    src_image_path = os.path.join(images_folder, image_file)
    dst_image_path = os.path.join(val_images_folder, image_file)
    shutil.copy(src_image_path, dst_image_path)

    # 复制对应的标签文件
    label_file = os.path.splitext(image_file)[0] + ".txt"
    src_label_path = os.path.join(data_folder, "labels", label_file)
    dst_label_path = os.path.join(val_labels_folder, label_file)
    shutil.copy(src_label_path, dst_label_path)


print("数据分割完成。")