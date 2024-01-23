import os

def rename_files_in_directory(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  # 确保处理的是JPG图片
            new_filename = filename.lstrip('0')  # 移除前导零
            if new_filename == '':
                new_filename = '0.jpg'  # 如果文件名全部由零组成，则重命名为0.jpg
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))  # 重命名文件

# 指定需要处理的文件夹路径
folder_path = r'/home/pmh/nvme1/Code/VBTS/data/pic'
rename_files_in_directory(folder_path)