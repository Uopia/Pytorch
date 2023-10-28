#删除末尾是_1的图片
# import os

# # 指定文件夹路径
# folder_path = r'D:\Desktop\data'

# # 遍历文件夹
# for root, dirs, files in os.walk(folder_path):
#     for filename in files:
#         if filename.endswith(("_1.jpg", "_1.png")):
#             # 构建完整文件路径
#             file_path = os.path.join(root, filename)
#             try:
#                 # 删除文件
#                 os.remove(file_path)
#                 print(f"已删除文件: {file_path}")
#             except Exception as e:
#                 print(f"删除文件时发生错误: {e}")

# print("删除操作完成。")



# 重命名文件
# import os

# # 指定文件夹路径
# folder_path = r'D:\Desktop\data'

# # 初始化计数器
# counter_dict = {}

# # 遍历文件夹
# for root, dirs, files in os.walk(folder_path):
#     for filename in files:
#         if filename.lower().endswith((".jpg", ".png")):
#             # 获取文件扩展名
#             file_extension = os.path.splitext(filename)[1]

#             # 获取当前子文件夹名称
#             folder_name = os.path.basename(root)

#             # 如果子文件夹名称不在计数器字典中，则初始化计数器为1
#             if folder_name not in counter_dict:
#                 counter_dict[folder_name] = 1

#             # 构建新文件名
#             new_filename = f"{folder_name}{counter_dict[folder_name]}{file_extension}"
            
#             # 构建完整文件路径
#             file_path = os.path.join(root, filename)
#             new_file_path = os.path.join(root, new_filename)
            
#             try:
#                 # 重命名文件
#                 os.rename(file_path, new_file_path)
#                 print(f"已重命名文件: {filename} -> {new_filename}")
#                 counter_dict[folder_name] += 1
#             except Exception as e:
#                 print(f"重命名文件时发生错误: {e}")

# print("重命名操作完成。")




## 对比两个文件夹中的文件名差异
# import os

# # 定义两个文件夹的路径
# folder1_path = r'D:\Desktop\Data1027\images'
# folder2_path = r'D:\Desktop\Data1027\labels'

# # 获取文件夹1中的文件名（不包含后缀）
# folder1_files = set(os.path.splitext(f)[0] for f in os.listdir(folder1_path) if os.path.isfile(os.path.join(folder1_path, f)))

# # 获取文件夹2中的文件名（不包含后缀）
# folder2_files = set(os.path.splitext(f)[0] for f in os.listdir(folder2_path) if os.path.isfile(os.path.join(folder2_path, f)))

# # 找到文件夹1中独特的文件名
# unique_to_folder1 = folder1_files - folder2_files

# # 找到文件夹2中独特的文件名
# unique_to_folder2 = folder2_files - folder1_files

# # 打印独特文件名
# print("文件夹1中独特的文件名:")
# for filename in unique_to_folder1:
#     print(filename)

# print("\n文件夹2中独特的文件名:")
# for filename in unique_to_folder2:
#     print(filename)



# # 删除15和8 更改16和17
# import os

# # 指定包含txt文件的文件夹路径
# folder_path = r'D:\Desktop\data\labels'

# # 遍历文件夹中的所有txt文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)
#         lines = []
#         with open(file_path, 'r') as file:
#             for line in file:
#                 # 分割每行的内容
#                 parts = line.strip().split()
#                 if len(parts) > 0:
#                     try:
#                         first_column = int(parts[0])
#                         if first_column not in [8, 15]:
#                             # 如果第一列不包含8和15，保留该行
#                             if first_column in [16, 17]:
#                                 # 如果第一列是16或17，替换为15或16
#                                 parts[0] = str(first_column - 1)
#                             lines.append(' '.join(parts))
#                     except ValueError:
#                         # 忽略无法转换为整数的行
#                         pass

#         # 保存处理后的内容回原文件
#         with open(file_path, 'w') as file:
#             for line in lines:
#                 file.write(line + '\n')


## 更改文件名TL
# import os

# # 指定包含文件的文件夹路径
# folder_path = r'D:\Desktop\data\images'

# # 获取文件夹中所有文件
# files = os.listdir(folder_path)

# # 初始化计数器
# counter = 1

# # 遍历文件夹中的文件
# for filename in files:
#     if "TL" in filename:
#         # 拆分文件名和扩展名
#         name, ext = os.path.splitext(filename)
        
#         # 构建新的文件名
#         new_filename = f"TL{counter}{ext}"
        
#         # 构建文件的完整路径
#         file_path = os.path.join(folder_path, filename)
#         new_file_path = os.path.join(folder_path, new_filename)
        
#         # 确保新文件名不会覆盖已存在的文件
#         while os.path.exists(new_file_path):
#             counter += 1
#             new_filename = f"TL{counter}{ext}"
#             new_file_path = os.path.join(folder_path, new_filename)
        
#         # 重命名文件
#         os.rename(file_path, new_file_path)
        
#         # 增加计数器
#         counter += 1

# print("重命名完成")



## 删除TL
# import os

# # 指定包含txt文件的文件夹路径
# folder_path = r'D:\Desktop\data\labels'

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if "TL" in filename:
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path):
#             # 删除文件
#             os.remove(file_path)
#             print(f"已删除文件: {filename}")

# print("删除操作完成")





# # 重命名文件
# import os

# # 指定包含txt文件的文件夹路径
# folder_path = r'D:\Desktop\data\labels'

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         # 拆分文件名和扩展名
#         name, ext = os.path.splitext(filename)
        
#         # 构建新的文件名
#         new_filename = f"{name}_1{ext}"
        
#         # 构建文件的完整路径
#         file_path = os.path.join(folder_path, filename)
#         new_file_path = os.path.join(folder_path, new_filename)
        
#         # 确保新文件名不会覆盖已存在的文件
#         while os.path.exists(new_file_path):
#             name = f"{name}_1"
#             new_filename = f"{name}{ext}"
#             new_file_path = os.path.join(folder_path, new_filename)
        
#         # 重命名文件
#         os.rename(file_path, new_file_path)
#         print(f"已重命名文件: {filename} 为 {new_filename}")

# print("重命名操作完成")



#  # 重命名文件
# import os

# # 指定包含文件的文件夹路径
# folder_path = r'D:\Desktop\data\images'

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 拆分文件名和扩展名
#     name, ext = os.path.splitext(filename)
    
#     # 构建新的文件名
#     new_filename = f"{name}_1{ext}"
    
#     # 构建文件的完整路径
#     file_path = os.path.join(folder_path, filename)
#     new_file_path = os.path.join(folder_path, new_filename)
    
#     # 确保新文件名不会覆盖已存在的文件
#     while os.path.exists(new_file_path):
#         name = f"{name}_1"
#         new_filename = f"{name}{ext}"
#         new_file_path = os.path.join(folder_path, new_filename)
    
#     # 重命名文件
#     os.rename(file_path, new_file_path)
#     print(f"已重命名文件: {filename} 为 {new_filename}")

# print("重命名操作完成")




# 划分数据集
import os
import random
import shutil

# 定义源文件夹路径
data_folder = "D:/Desktop/Data1027"

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


    

# print("数据分割完成。")

# 统计每个类别的行数
# import os
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties

# # 文件夹路径
# folder_path = r'D:\Desktop\Data1027\labels'

# # 用于存储每个类别的行数的字典
# class_counts = {str(i): 0 for i in range(18)}  # 初始化每个类别的行数为0
# total_lines = 0  # 用于存储总共多少行数据

# # 定义不同类别的颜色
# colors = ['g', 'c', 'y']

# # 遍历文件夹中的每个txt文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#             total_lines += len(lines)  # 更新总行数
#             for line in lines:
#                 # 切分每行数据
#                 data = line.strip().split()
#                 if len(data) > 0:
#                     # 获取类别
#                     class_label = data[0]
#                     # 更新字典中该类别的行数
#                     class_counts[class_label] += 1

# # 打印每个类别的行数
# for class_label in sorted(class_counts.keys(), key=int):
#     count = class_counts[class_label]
#     print(f'类别 {class_label}: {count} 行')

# # 打印总共多少行数据
# print(f'总共 {total_lines} 行数据')

# # 创建图表
# fig, ax = plt.subplots()
# bars = plt.bar(class_counts.keys(), [class_counts[str(i)] for i in range(18)], color=colors[:18])

# # 添加标签和数字（调整数字位置）
# for bar, count in zip(bars, [class_counts[str(i)] for i in range(18)]):
#     plt.text(bar.get_x() + bar.get_width() / 2 - 0.5, bar.get_height() + 0.5, str(count), fontsize=10)

# # 设置中文字体
# font_properties = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 请根据你的系统中文字体文件路径进行设置

# plt.xlabel('类别', fontproperties=font_properties)
# plt.ylabel('行数', fontproperties=font_properties)
# plt.title('类别行数统计', fontproperties=font_properties)
# plt.show()



#删同名文件
# import os
# import glob

# # 文件夹A和B的路径
# folder_a = r'D:\Desktop\Data1027\images'
# folder_b = r'D:\Desktop\Data1027\page_b'

# # 获取文件夹B中的文件名（不包括扩展名）
# files_b = set([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(os.path.join(folder_b, '*.*'))])

# # 遍历文件夹A中的文件
# for root, _, files_a in os.walk(folder_a):
#     for file_a in files_a:
#         # 获取文件名（不包括扩展名）
#         filename_a = os.path.splitext(file_a)[0]
#         # 如果文件名在文件夹B中存在，则删除文件夹A中的文件
#         if filename_a in files_b:
#             file_a_path = os.path.join(root, file_a)
#             os.remove(file_a_path)
#             print(f"已删除文件: {file_a_path}")

# print("完成删除操作")


# import os

# # 文件夹A和B的路径
# folder_a = r'D:\Desktop\Data1027\images'
# folder_b = r'D:\Desktop\Data1027\labels'

# # 获取文件夹A中的图片文件名（不包括扩展名）
# image_files_a = [os.path.splitext(file)[0] for file in os.listdir(folder_a) if file.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tif'))]

# # 获取文件夹B中的txt文件名（不包括扩展名）
# txt_files_b = [os.path.splitext(file)[0] for file in os.listdir(folder_b) if file.lower().endswith('.txt')]

# # 找出B中没有对应图片文件的txt文件名
# files_to_remove = [file_b for file_b in txt_files_b if file_b not in image_files_a]

# # 遍历B中要删除的txt文件名，删除对应的txt文件
# for file_to_remove in files_to_remove:
#     file_b_path = os.path.join(folder_b, file_to_remove + '.txt')
#     if os.path.exists(file_b_path):
#         os.remove(file_b_path)
#         print(f"已删除文件: {file_b_path}")

# print("完成删除操作")


# import os

# # 文件夹路径
# folder_path = r'D:\Desktop\Data1027\images'

# # 用于存储文件名和对应文件路径的字典
# file_dict = {}

# # 遍历文件夹中的文件
# for root, _, files in os.walk(folder_path):
#     for file in files:
#         file_name, file_ext = os.path.splitext(file)
#         if file_name not in file_dict:
#             file_dict[file_name] = [os.path.join(root, file)]
#         else:
#             file_dict[file_name].append(os.path.join(root, file))

# # 查找文件名相同但后缀不同的文件
# for file_name, file_paths in file_dict.items():
#     if len(file_paths) > 1:
#         print(f'文件名相同但后缀不同的文件:')
#         for file_path in file_paths:
#             print(file_path)
