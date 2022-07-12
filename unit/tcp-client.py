import cv2
import time
import socket
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=6001)
args = parser.parse_args()
HOST = args.ip
PORT = args.port
ADDRESS = (HOST,PORT)
tcpClient = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcpClient.connect(ADDRESS)

cap = cv2.VideoCapture(0)
ref,cv_image = cap.read()
print("{}".format(cv_image.shape))
while True:
    start = time.perf_counter()
    ref,cv_image = cap.read()
    # cv2.imshow("client",cv_image)

    # img_encode = cv2.imencode('.jpg',cv_image,[cv2.IMWRITE_JPEG_QUALITY,99])[1]
    # bytedata = img_encode.tobytes()

    bytedata = cv_image.tobytes()
    flag_data = (str(len(bytedata))).encode()+"..".encode()+" ".encode()
    tcpClient.send(flag_data)
    data = tcpClient.recv(1024)
    if("ok"==data.decode()):
        tcpClient.send(bytedata)
    data = tcpClient.recv(1024)
    if("ok"==data.decode()):
        #print("time:{}ms".format((time.perf_counter()-start)*1000))
        pass
    if(cv2.waitKey(10)=="q"):
        break


    data = tcpClient.recv(1024)
    if data:
        tcpClient.send(b"ok")
        flag = data.decode().split("..")
        total = int(flag[0])
        cnt = 0
        img_bytes = b''
        while cnt < 640*480*3:
            data = tcpClient.recv(1024)
            img_bytes += data
            cnt += len(data)
            # print("receive:{}/{}\r".format(cnt,flag[0]))
        tcpClient.send(b"ok")
        if len(img_bytes) != 640*480*3:
            continue            
        img = np.asarray(bytearray(img_bytes),dtype="uint8")
        img = img.reshape((480,640,3))
        cv2.imshow("re call",img)
