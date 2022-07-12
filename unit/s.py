from ipaddress import v4_int_to_packed
import cv2
import socket
import numpy as np
import yolov5_dnn
from 检测图片 import detect

HOST = "127.0.0.1"
PORT = 6001
ADDRESS = (HOST,PORT)
tcpServer = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcpServer.bind(ADDRESS)
tcpServer.listen(5)
model = yolov5_dnn.yolov5(r'./weights/yolov5n.onnx')
while True:
    print("wait for connect\n")
    client_socket,client_address = tcpServer.accept()
    print("connect success\n")
    while True:
        data = client_socket.recv(1024)
        if data:
            client_socket.send(b"ok")
            flag = data.decode().split("..")
            total = int(flag[0])
            cnt = 0
            img_bytes = b''
            while cnt < 640*480*3:
                data = client_socket.recv(1024)
                img_bytes += data
                cnt += len(data)
            client_socket.send(b"ok")
            if len(img_bytes) != 640*480*3:
                continue
            img = np.asarray(bytearray(img_bytes),dtype="uint8")
            img = img.reshape((480,640,3))
            # print(img.shape)
            # img = cv2.imdecode(img,cv2.IMREAD_COLOR)

            #图片 处理 开始
            img = model.detect(img)
            cv2.imshow('result', img)
            cv2.waitKey(1)
            #图片 处理 结束

            #图片 回传 开始
            bytedata = img.tobytes()
            flag_data = (str(len(bytedata))).encode()+"..".encode()+" ".encode()
            client_socket.send(flag_data)
            data = client_socket.recv(1024)
            if("ok"==data.decode()):
                client_socket.send(bytedata)
            data = client_socket.recv(1024)
            if("ok"==data.decode()):
                #print("time:{}ms".format((time.perf_counter()-start)*1000))
                pass
            cv2.imshow("server",img)
            cv2.waitKey(10)
            #图片 回传 结束
        else:
            print("connect close")
            break
    cv2.destroyAllWindows()
    client_socket.close()
