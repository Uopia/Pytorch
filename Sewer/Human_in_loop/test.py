# 第一步，手动遍历人在回路1
import cv2
import os

def get_screen_center(screen_resolution=(2560, 1440)):
    return screen_resolution[0] // 2, screen_resolution[1] // 2

def resize_image(image, max_window_size=(1000, 800)):
    height, width = image.shape[:2]
    scale_width = max_window_size[0] / width
    scale_height = max_window_size[1] / height
    scale = min(scale_width*2, scale_height*2)

    window_width = int(width * scale)
    window_height = int(height * scale)

    return cv2.resize(image, (window_width, window_height)), (window_width, window_height)

def show_image_with_text(image_path, screen_center, max_window_size):
    image = cv2.imread(image_path)
    resized_image, (window_width, window_height) = resize_image(image, max_window_size)

    filename = os.path.basename(image_path)
    cv2.putText(resized_image, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    window_x = screen_center[0] - window_width // 2
    window_y = screen_center[1] - window_height // 2

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', window_width, window_height)
    cv2.moveWindow('Image', window_x, window_y)
    cv2.imshow('Image', resized_image)

def rename_associated_txt_file(path, image_name, new_image_name):
    old_txt_name = image_name.split('.')[0] + '.txt'
    new_txt_name = new_image_name.split('.')[0] + '.txt'

    old_txt_path = os.path.join(path, old_txt_name)
    new_txt_path = os.path.join(path, new_txt_name)

    if os.path.exists(old_txt_path):
        os.rename(old_txt_path, new_txt_path)
        return new_txt_name
    return None

def main():
    path = r"D:\Desktop\lab_g\ZW"
    images = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    current_index = 0
    deleted_images = []
    renamed_images = []
    screen_center = get_screen_center()
    max_window_size = (1000, 800)

    while current_index < len(images):
        image_path = os.path.join(path, images[current_index])
        show_image_with_text(image_path, screen_center, max_window_size)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'):
            os.remove(image_path)
            deleted_images.append(images[current_index])
            del images[current_index]
            # 修正了删除操作后的 current_index 处理
            if current_index >= len(images):
                break
            continue  # 继续显示当前索引的图片
        elif key == ord('o'):
            current_index = max(0, current_index - 1)
            continue
        elif key == ord('p'):
            current_index = min(len(images) - 1, current_index + 1)
            continue
        elif key == ord('e'):
            new_image_name = 'a' + images[current_index]
            new_image_path = os.path.join(path, new_image_name)
            os.rename(image_path, new_image_path)
            renamed_images.append(new_image_name)
            renamed_txt = rename_associated_txt_file(path, images[current_index], new_image_name)
            if renamed_txt:
                print(f"Renamed TXT file: {renamed_txt}")
            images[current_index] = new_image_name

        elif key == 27:  # ESC 键退出
            break
        
        current_index += 1

    cv2.destroyAllWindows()
    print("Deleted images:", deleted_images)
    print("Renamed images:", renamed_images)

if __name__ == "__main__":
    main()
