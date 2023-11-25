# 第三步，将labels文件夹中的所有.txt文件对应的图片复制到指定文件夹中
import os
import shutil

def find_and_copy_image(txt_filename, images_folder, dest_folder):
    # 构造基础的图片文件名（去除.txt前缀和扩展名）
    base_image_name = txt_filename[1:].replace('.txt', '')

    # 支持的图片格式列表
    supported_formats = ['.png', '.jpg', '.jpeg']

    for format in supported_formats:
        image_name = base_image_name + format
        image_path = os.path.join(images_folder, image_name)

        # 如果找到匹配的图片，则复制并返回
        if os.path.exists(image_path):
            new_image_name = 'a' + image_name
            dest_image_path = os.path.join(dest_folder, new_image_name)
            shutil.copy(image_path, dest_image_path)
            print(f'Copied: {image_path} -> {dest_image_path}')
            return True
    
    # 如果没有找到匹配的图片
    return False

def copy_matching_images(labels_folder, images_folder, dest_folder):
    # 确保目标文件夹存在
    os.makedirs(dest_folder, exist_ok=True)

    # 遍历labels文件夹中的所有.txt文件
    for filename in os.listdir(labels_folder):
        if filename.lower().endswith('.txt'):
            if not find_and_copy_image(filename, images_folder, dest_folder):
                print(f'No matching image found for {filename}')

# 设置文件夹路径
labels_folder = 'D:\\Desktop\\Data11\\lab_g\\AJ'
images_folder = 'D:\\Desktop\\data\\images'
dest_folder = 'D:\\Desktop\\images'

# 执行复制操作
copy_matching_images(labels_folder, images_folder, dest_folder)
