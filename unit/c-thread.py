import atexit
import cv2
import time
import socket
import numpy as np
import argparse
import threading
from numpy import random

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=6001)
args = parser.parse_args()
HOST = args.ip
PORT = args.port


bALL_EXIT = False

lock_received = threading.Lock() ### 画框锁 当锁定时，接收到新的图像框
receive_rectangle = list()
receive_rectangle.clear()

lock_sent = threading.Lock() ### 图片锁 当锁定时，正在准备发送图片
bytedata = b""

bSendFlag = False   ### 发送标志 当为真时，表示可以发送图片

def thread_received():  ### 图片发送接收线程
    global receive_rectangle,bytedata,bSendFlag,bALL_EXIT
    try:        
        print("thread successfully started:connect")
        print("bSendFlag :{}".format(bSendFlag))
        ADDRESS = (HOST,PORT)
        tcpClient = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        tcpClient.connect(ADDRESS)
        while True:
            if bALL_EXIT:            
                break
            if bSendFlag:
                bSendFlag = False
                lock_sent.acquire() 
                # print("Sent message image started")
                tcpClient.send(bytedata)
                # print("Sent message image successfully")
                lock_sent.release()

                temp = list()
                data = tcpClient.recv(1024)
                data = list(data)
                # print(data)
                for i in range(0,int(len(data)/8)):
                    temp.append(  list([ (data[__*2+8*i] )*255+data[__*2+1+8*i] for __ in range(0,4) ]) )
                lock_received.acquire() ### 画框锁 上锁
                receive_rectangle.clear()
                receive_rectangle = temp
                # print(receive_rectangle)
                lock_received.release() ### 画框锁 解锁
    finally:
        bALL_EXIT = True
        print("connect exit")


def thread_main():
    global receive_rectangle,bytedata,bSendFlag,bALL_EXIT
    try:        
        print("thread successfully started:main")
        print("bSendFlag :{}".format(bSendFlag))

        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source' #摄像头没有正常打开则报错
        ref,cv_image = cap.read()
        print("{}".format(cv_image.shape))
        cv2.imshow("recall",cv_image)
        cv2.waitKey(1)    
        colors = [random.randint(0, 255) for _ in range(3)]     # 设置画框的颜色  #######################
        print(colors)

        while True:
            if bALL_EXIT:            
                break
            ### send start
            ref,cv_image = cap.read()
            # cv2.flip(cv_image,-1,cv_image)
            if bSendFlag==False:
                lock_sent.acquire() 
                bytedata = cv_image.tobytes()
                bSendFlag = True
                # print("ready to send data")
                lock_sent.release()
            ### send end

            if(len(receive_rectangle)!=0):
                for _ in receive_rectangle:
                    cv2.rectangle(cv_image, (_[0], _[1]), (_[2], _[3]), colors)
            cv2.imshow("recall",cv_image)
            cv2.waitKey(1)
        cap.release()
    finally:
        bALL_EXIT = True
        print("main exit")


threads = list()
threads.append(threading.Thread(target=thread_main))
threads[-1].start()
threads.append(threading.Thread(target=thread_received))
threads[-1].start()

@atexit.register
def __clean():
    global bALL_EXIT
    bALL_EXIT = True
    print("clean all exit")


for t in threads:
    t.join()
print("exit successfully")
