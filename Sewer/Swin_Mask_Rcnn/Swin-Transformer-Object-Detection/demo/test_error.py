import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from tqdm import tqdm  # 导入tqdm

config = r'D:\Desktop\Swin\Swin-Transformer-Object-Detection\work_dirs\mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint = r'D:\Desktop\Swin\Swin-Transformer-Object-Detection\work_dirs\mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth'

device = 'cuda:0'

# 指定待检测的文件夹路径
img_folder = r'D:\Desktop\1'  # 替换成你的文件夹路径
model = init_detector(config, checkpoint, device=device)
# 获取文件夹中所有图像文件的路径
img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

# 使用tqdm包装图像路径列表以显示进度
for img_path in tqdm(img_paths, desc="Processing images"):
    try:
        # 运行目标检测函数
        results = inference_detector(model, img_path)
    except Exception as e:
        print(f"Error detected in image: {img_path}")
        print(f"Error message: {e}")


