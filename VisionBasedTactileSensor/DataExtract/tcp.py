import socket
import threading

def receive_data(client_socket):
    while True:
        try:
            response = client_socket.recv(1024).decode()
            if not response:
                print("Server disconnected")
                break
            print("Received from server:", response)
        except Exception as e:
            print("Error receiving data:", e)
            break

def client_program():
    host = '169.254.160.100'
    port = 502

    client_socket = socket.socket()
    client_socket.connect((host, port))
    print("Connected to server at {}:{}".format(host, port))

    # 开始接收数据的线程
    threading.Thread(target=receive_data, args=(client_socket,), daemon=True).start()

    try:
        while True:
            data = input("Enter data to send: ")
            if data.lower() == 'quit':
                break
            client_socket.sendall(data.encode())
    except KeyboardInterrupt:
        pass
    finally:
        client_socket.close()
        print("Connection closed")

if __name__ == '__main__':
    client_program()
