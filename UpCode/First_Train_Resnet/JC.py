# import os
# import random
# import cv2
# import numpy as np
#
# folder_path = r'F:\Data\Sewer\sumT'
# num_images_to_display = 10
# window_name = "Random Images"
#
# # 获取文件夹中的所有图片文件
# image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
#
# # 随机选择要显示的图片
# selected_images = random.sample(image_files, num_images_to_display)
#
# # 创建一个用于显示图片的大画布
# canvas = np.zeros((500, 1000, 3), dtype=np.uint8)
#
# # 将图片填充到画布上
# x, y = 50, 50
# width, height = 100, 100
#
# for i, image_file in enumerate(selected_images):
#     image_path = os.path.join(folder_path, image_file)
#     image = cv2.imread(image_path)
#
#     # 调整图片大小以适应指定的宽度和高度
#     image = cv2.resize(image, (width, height))
#
#     # 将图片放置在画布上的指定位置
#     canvas[y:y+height, x:x+width] = image
#
#     # 在图片下方添加图片名字标签
#     cv2.putText(canvas, image_file, (x, y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#
#     # 更新下一个图片的位置
#     x += width + 50
#
#     # 每行显示5张图片
#     if (i + 1) % 5 == 0:
#         x = 50
#         y += height + 100
#
# # 显示图片
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.imshow(window_name, canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import os
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
csv_file_path = r'F:\Data\Sewer\SewerML_Train.csv'
data = pd.read_csv(csv_file_path)

# 获取标签名和类别列
label_names = data.columns[2:]
label_columns = data[label_names]

# 统计每个类别的图片数量
class_counts = label_columns.sum()

# 可视化结果
plt.bar(label_names, class_counts)
plt.xlabel('Class')
plt.ylabel('Image Count')
plt.title('Image Count per Class')
plt.xticks(rotation=90)
plt.show()
