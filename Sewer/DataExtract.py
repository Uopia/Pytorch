# 1 合并两个数据集的图片
import os
import shutil


def get_new_name(path, filename):
    """为重名文件生成新的文件名"""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_name = f"{base}_{counter}{ext}"
    while os.path.exists(os.path.join(path, new_name)):
        counter += 1
        new_name = f"{base}_{counter}{ext}"
    return new_name


# 源文件夹路径
source_folder_path = r"F:\Data\QV图片标注"
# 目标文件夹路径
destination_folder_path = r"F:\\Data\\QV图片标注"

# 遍历源文件夹中的所有子文件夹
for subfolder_name in os.listdir(source_folder_path):
    subfolder_path = os.path.join(source_folder_path, subfolder_name)

    # 检查是否为子文件夹
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹中的所有孙文件夹
        for grandson_folder_name in os.listdir(subfolder_path):
            grandson_folder_path = os.path.join(subfolder_path, grandson_folder_name)

            # 检查是否为孙文件夹
            if os.path.isdir(grandson_folder_path):
                # 创建目标文件夹路径
                target_folder_path = os.path.join(
                    destination_folder_path, grandson_folder_name
                )

                # 如果目标文件夹不存在，则创建它
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)

                # 复制孙文件夹中的所有文件到目标文件夹
                for filename in os.listdir(grandson_folder_path):
                    source_file_path = os.path.join(grandson_folder_path, filename)
                    destination_file_path = os.path.join(target_folder_path, filename)

                    # 如果文件已存在，则生成新的文件名
                    if os.path.exists(destination_file_path):
                        filename = get_new_name(target_folder_path, filename)
                        destination_file_path = os.path.join(
                            target_folder_path, filename
                        )

                    shutil.copy2(source_file_path, destination_file_path)

print("合并完成！")

# 2 删除非png文件
# import os

# def remove_non_image_files(folder_path):
#     """递归删除非jpg和png文件"""
#     for root, dirs, files in os.walk(folder_path, topdown=False):
#         # 如果没有子目录，则认为是最底层目录
#         if not dirs:
#             for filename in files:
#                 if not (filename.endswith('.jpg') or filename.endswith('.png')):
#                     file_path = os.path.join(root, filename)
#                     os.remove(file_path)
#                     print(f"已删除文件：{file_path}")

# # 文件夹路径
# folder_path = "F:\\Data\\QV图片标注\\1\\1.1、图片标注"

# remove_non_image_files(folder_path)

# print("清除完成！")



# 3 去除正常行
# import openpyxl
# import os

# # 文件夹路径和Excel文件路径
# folder_path = "F:\\Data\\QV图片标注\\2\\1217"
# excel_file_path = os.path.join(folder_path, "20211217.xlsx")
# output_file_path = os.path.join(folder_path, "output.xlsx")

# # 打开Excel文件
# workbook = openpyxl.load_workbook(excel_file_path)
# sheet = workbook.active

# # 创建一个新的工作簿和工作表
# new_workbook = openpyxl.Workbook()
# new_sheet = new_workbook.active

# # 遍历原工作表的所有行
# row_num = 1
# for row in sheet.iter_rows():
#     # 检查第一列的值是否为"正常"
#     if row[0].value != "正常":
#         # 将非"正常"的行复制到新的工作表中
#         for col_num, cell in enumerate(row, start=1):
#             new_sheet.cell(row=row_num, column=col_num, value=cell.value)
#         row_num += 1

# # 保存新的Excel文件
# new_workbook.save(output_file_path)

# print(f"已保存新的Excel文件到 {output_file_path}")


# 4 将名称换成首字母大写
# import openpyxl
# import os

# # 定义汉字到字符的映射
# mapping = {
#     "支管暗接": "AJ",
#     "变形": "BX",
#     "沉积": "CJ",
#     "异物穿入": "CR",
#     "错口": "CK",
#     "残墙、坝根": "CQ",
#     "腐蚀": "FS",
#     "浮渣": "FZ",
#     "结垢": "JG",
#     "破裂": "PL",
#     "起伏": "QF",
#     "树根": "SG",
#     "渗漏": "SL",
#     "接口材料脱落": "TL",
#     "障碍物": "ZW",
#     "脱节": "TJ",
# }

# # Excel文件路径
# excel_file_path = "F:\\Data\\QV图片标注\\2\\1217\\output.xlsx"

# # 打开Excel文件
# workbook = openpyxl.load_workbook(excel_file_path)
# sheet = workbook.active

# # 遍历工作表中的所有行
# for row in sheet.iter_rows():
#     # 获取第一列的值
#     value = row[0].value
#     # 如果该值在映射中，则替换为对应的字符
#     if value in mapping:
#         row[0].value = mapping[value]

#  保存修改后的Excel文件
# workbook.save(excel_file_path)

# print(f"已修改Excel文件 {excel_file_path}")


# 5 提取图片
# import os
# import openpyxl
# import shutil

# # 文件夹和Excel文件路径
# image_folder_path = "F:\\Data\\QV图片标注\\2\\1217\\图片"
# excel_file_path = "F:\\Data\\QV图片标注\\2\\1217\\output.xlsx"
# output_folder_path = "F:\\Data\\QV图片标注\\2\\1217\\label"

# # 打开Excel文件
# workbook = openpyxl.load_workbook(excel_file_path)
# sheet = workbook.active

# # 初始化计数器
# moved_count = 0
# total_images = len(
#     [
#         name
#         for name in os.listdir(image_folder_path)
#         if name.endswith(".jpg") or name.endswith(".png")
#     ]
# )

# # 遍历工作表中的所有行
# for row in sheet.iter_rows(min_row=2):  # 假设第一行是标题行，从第二行开始遍历
#     code = row[0].value
#     image_name_without_ext = row[1].value

#     # 检查.jpg和.png后缀的图片是否存在
#     for ext in [".jpg", ".png"]:
#         image_path = os.path.join(image_folder_path, image_name_without_ext + ext)
#         if os.path.exists(image_path):
#             # 创建对应的分类文件夹
#             target_folder = os.path.join(output_folder_path, code)
#             if not os.path.exists(target_folder):
#                 os.makedirs(target_folder)

#             # 复制图片到分类文件夹中
#             shutil.copy2(
#                 image_path, os.path.join(target_folder, image_name_without_ext + ext)
#             )
#             moved_count += 1
#             print(f"已复制 {moved_count} 张图片，还剩 {total_images - moved_count} 张。")
#             break

# print("分类完成！")
