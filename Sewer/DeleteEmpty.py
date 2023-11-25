import os

# 设置文件夹路径
folder_path = r'D:\Desktop\Data11\labels'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 确保是文件且为txt文件
    if os.path.isfile(file_path) and filename.endswith('.txt'):
        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            print(f"删除空文件: {filename}")
            os.remove(file_path)
