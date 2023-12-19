# 对比两个文件夹中的文件名差异
import os

# 定义两个文件夹的路径
folder1_path = r'D:\Desktop\Data\images'
folder2_path = r'D:\Desktop\Data\labels'

# 获取文件夹1中的文件名（不包含后缀）
folder1_files = set(os.path.splitext(f)[0] for f in os.listdir(folder1_path) if os.path.isfile(os.path.join(folder1_path, f)))

# 获取文件夹2中的文件名（不包含后缀）
folder2_files = set(os.path.splitext(f)[0] for f in os.listdir(folder2_path) if os.path.isfile(os.path.join(folder2_path, f)))

# 找到文件夹1中独特的文件名
unique_to_folder1 = folder1_files - folder2_files

# 找到文件夹2中独特的文件名
unique_to_folder2 = folder2_files - folder1_files

# 打印独特文件名
print("文件夹1中独特的文件名:")
for filename in unique_to_folder1:
    print(filename)

print("\n文件夹2中独特的文件名:")
for filename in unique_to_folder2:
    print(filename)


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