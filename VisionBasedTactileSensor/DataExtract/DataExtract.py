import socket
import threading
import cv2
import csv
import serial
from datetime import datetime

# 全局变量
tcp_data = []
# 全局摄像头对象
cap = None


def receive_data(client_socket):
    while True:
        try:
            response = client_socket.recv(1024).decode()
            if not response:
                print("Server disconnected")
                break
            tcp_data.append(response)
        except Exception as e:
            print("Error receiving data:", e)
            break

def save_to_csv(data, filename):
    try:
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if len(data) > 3:
                data = data[:3]  # 只保留前三个数据
            writer.writerow(data)
    except Exception as e:
        print(f"Error writing to CSV: {e}")


def capture_image(number):
    global cap
    ret, frame = cap.read()
    if ret:
        filename = f"D:/Desktop/pic/{number:03d}.jpg"  # 使用number作为文件名
        cv2.imwrite(filename, frame)
    else:
        print("无法从摄像头获取画面")


def continuous_capture():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOCUS,999)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera View', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("无法从摄像头获取画面")
            break

def client_program():
    host = '169.254.160.100'  # 服务器IP
    port = 502               # 服务器端口
    client_socket = socket.socket()
    client_socket.connect((host, port))
    print("Connected to server at {}:{}".format(host, port))
    threading.Thread(target=receive_data, args=(client_socket,), daemon=True).start()

def read_serial():
    try:
        print("read_serial thread started")
        ser = serial.Serial('COM3', 9600)  # 串口号
        last_number = -1  # 用于存储上一次读取的数据
        loop_counter = 0  # 循环计数器

        while loop_counter < 300:
            data = ser.read_until(b'S')  # 读取直到遇到 'S'
            data_str = data.decode('utf-8').strip('S').strip()  # 去除 'S' 和空白字符
            print("Received:", data_str)
            if data_str == "tart":
                ser.write("okS".encode())  # 发送 "okS"
            elif data_str.startswith('A'):
                number_str = data_str.strip('A')
                try:
                    number = int(number_str)  # 解析数字
                    if number == last_number:
                        print("error: Same data received as last loop")
                    else:
                        print("Received number:", number)
                        capture_image(number)
                        save_to_csv([number] + tcp_data[:2], "D:/Desktop/data.csv")
                        ser.write("okS".encode())  # 发送 "okS"
                        last_number = number  # 更新 last_number
                        loop_counter += 1
                except ValueError:
                    print("Error: Received data is not a valid number")
            else:
                print("No valid command received")

    except Exception as e:
        print(f"Error in read_serial: {e}")


def main():

    camera_thread = threading.Thread(target=continuous_capture, daemon=True)
    client_thread = threading.Thread(target=client_program, daemon=True)
    serial_thread = threading.Thread(target=read_serial, daemon=True)

    camera_thread.start()
    client_thread.start()
    serial_thread.start()

    camera_thread.join()
    client_thread.join()
    serial_thread.join()

if __name__ == '__main__':
    main()
