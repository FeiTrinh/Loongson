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
    ### send start
    start = time.perf_counter()
    ref,cv_image = cap.read()
    cv2.flip(cv_image,-1,cv_image)
    bytedata = cv_image.tobytes()
    print("capture time:{}ms".format((time.perf_counter()-start)*1000))
    start = time.perf_counter()
    tcpClient.send(bytedata)
    print("send time:{}ms".format((time.perf_counter()-start)*1000))
    ### send end

    ### rece start
    start = time.perf_counter()
    cnt = 0
    img_bytes = b''
    while cnt < 640*480*3:
        data = tcpClient.recv(1024)
        img_bytes += data
        cnt += len(data)
    print("rece time:{}ms".format((time.perf_counter()-start)*1000))
    if len(img_bytes) != 640*480*3:
        continue            
    img = np.asarray(bytearray(img_bytes),dtype="uint8")
    start = time.perf_counter()
    ### rece end
    img = img.reshape((480,640,3))
    cv2.imshow("re call",img)
    print("show time:{}ms".format((time.perf_counter()-start)*1000))
