# 第四步，把两个文件夹中的文件移动到一个文件夹中，并重命名
import os
import shutil

def move_and_rename_files(src_folders, dest_folder):
    for src_folder in src_folders:
        for filename in os.listdir(src_folder):
            # 构建完整的文件路径
            src_file_path = os.path.join(src_folder, filename)

            # 检查文件名并去除开头的'a'
            if filename.startswith('a'):
                new_filename = filename[1:]
            else:
                new_filename = filename

            dest_file_path = os.path.join(dest_folder, new_filename)

            # 移动并重命名文件
            shutil.move(src_file_path, dest_file_path)
            print(f'Moved and renamed: {src_file_path} -> {dest_file_path}')

# 设置文件夹路径
base_folder = 'D:\\Desktop\\lab_g\\ZW'
images_folder = os.path.join(base_folder, 'images')
labels_folder = os.path.join(base_folder, 'labels')

# 执行操作
move_and_rename_files([images_folder, labels_folder], base_folder)












