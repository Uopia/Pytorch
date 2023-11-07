import cv2
import os

# 文件夹路径包含图像
image_folder = 'V2P'  # 指定包含图像的文件夹名称
output_video_folder = 'P2V'  # 指定保存视频的文件夹名称
output_video = 'output_video.mp4'  # 输出视频文件名

# 创建保存视频的文件夹
os.makedirs(output_video_folder, exist_ok=True)

# 获取图像文件列表
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 设置视频编解码器和帧速率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(output_video_folder, output_video)
video = cv2.VideoWriter(video_path, fourcc, 15, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

cv2.destroyAllWindows()
video.release()

print(f"已生成视频: {video_path}")
