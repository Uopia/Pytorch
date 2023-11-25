# 删除文件中多余的txt，未和jpg png对应的txt
# import os

# def filter_txt_files(folder_path):
#     # 获取文件夹下所有子文件夹的路径
#     subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

#     # 遍历每个子文件夹
#     for subfolder in subfolders:
#         jpg_files = set()
#         txt_files_to_keep = set()

#         # 遍历当前子文件夹下的所有文件
#         for file in os.scandir(subfolder):
#             # 如果是jpg文件，将文件名（不带后缀）添加到jpg_files集合
#             if file.name.lower().endswith('.jpg') and file.is_file():
#                 jpg_files.add(os.path.splitext(file.name)[0])
#             # 如果是txt文件，且文件名（不带后缀）在jpg_files集合中，将文件路径添加到txt_files_to_keep集合
#             elif file.name.lower().endswith('.txt') and file.is_file() and os.path.splitext(file.name)[0] in jpg_files:
#                 txt_files_to_keep.add(file.path)

#         # 删除不需要的txt文件
#         for file in os.scandir(subfolder):
#             if file.name.lower().endswith('.txt') and file.is_file() and file.path not in txt_files_to_keep:
#                 os.remove(file.path)

# if __name__ == "__main__":
#     # 替换为你的文件夹路径，到AJ、BX上一级即可
#     your_folder_path = r"D:\Desktop\Data11\lab_b"
    
#     filter_txt_files(your_folder_path)




# 将多个文件夹中的txt文件合并（移动和叠加labels）到一个文件夹中，并且删除多余的jpg文件
import os

def merge_and_move_txt_files_single(folder_a, folder_b):
    # 获取文件夹中的所有txt文件
    files_a = {f.name: f for f in os.scandir(folder_a) if f.is_file() and f.name.endswith('.txt')}
    files_b = [f for f in os.scandir(folder_b) if f.is_file() and f.name.endswith('.txt')]

    # 遍历文件夹b中的每个txt文件
    for file_b in files_b:
        file_b_name = file_b.name
        file_a_path = files_a.get(file_b_name)

        # 如果文件夹a中存在同名的txt文件
        if file_a_path:
            with open(file_a_path, 'a') as file_a:
                # 读取文件夹b中的内容并逐行写入文件夹a中的文件
                with open(file_b, 'r') as file_b_content:
                    for line in file_b_content:
                        file_a.write(line)
        else:
            # 如果文件夹a中不存在同名的txt文件，则将文件夹b中的txt文件移动到文件夹a中
            os.rename(file_b.path, os.path.join(folder_a, file_b_name))

    # 删除文件夹a中的jpg文件
    jpg_files_a = [f.path for f in os.scandir(folder_a) if f.is_file() and f.name.endswith('.jpg')]
    for jpg_file in jpg_files_a:
        os.remove(jpg_file)

if __name__ == "__main__":
    # 文件夹路径，到AJ、BX上一级即可
    source_folder = r"D:\Desktop\Data11"
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    # 按照指定顺序执行 merge_and_move_txt_files_single 操作
    for i in range(1, len(subfolders)):
        folder_a = os.path.join(source_folder, subfolders[0])
        folder_b = os.path.join(source_folder, subfolders[i])
        merge_and_move_txt_files_single(folder_a, folder_b)
