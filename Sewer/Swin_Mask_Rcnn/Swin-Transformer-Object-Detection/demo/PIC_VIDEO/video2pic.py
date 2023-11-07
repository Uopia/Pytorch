import cv2
import os

# 打开视频文件
video_path = '1.mp4'  # 将'your_video.mp4'替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 创建一个文件夹来保存图像
output_folder = 'V2P/1'  # 指定保存图像的文件夹名称
os.makedirs(output_folder, exist_ok=True)

# 设置输出图像的帧速率
output_frame_rate = 15  # 每秒15张图像
output_interval = int(1 / output_frame_rate * 1000)  # 以毫秒为单位的时间间隔

frame_count = 0
output_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # 检查是否应该输出当前帧
    if frame_count % (cap.get(cv2.CAP_PROP_FPS) // output_frame_rate) == 0:
        output_count += 1
        output_filename = os.path.join(output_folder, f"output_frame_{output_count:04d}.jpg")  # 图像文件名，保存在指定文件夹中
        cv2.imwrite(output_filename, frame)
        print(f"输出图像: {output_filename}")
    
    # 显示视频（可选）
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(output_interval) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
