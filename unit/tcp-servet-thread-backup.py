from ipaddress import v4_int_to_packed
from re import X
import cv2
import socket
import numpy as np
import torch
import time
from numpy import byte, random

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import plot_one_box

weights ='yolov5s.pt'   #######################
device ='cuda:0'   #######################
model = attempt_load(weights, map_location=device)  # load FP32 model   #######################

names = model.module.names if hasattr(model, 'module') else model.names  # 获取类别名字   #######################
print(names) # print
choice_labels = ['person']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]     # 设置画框的颜色  #######################

def detect(img):
    box = [0 for _ in range(8)]
    x_img = img
    im0 = img
    x_img = np.array(x_img, dtype=np.float32)
    x_img = x_img.transpose(2, 0, 1)
    x_img = torch.from_numpy(x_img).to(device)
    x_img /= 255.0  # 0 - 255 to 0.0 - 1.0

    x_img = x_img.unsqueeze(0)
    # Inference
    pred = model(x_img)
    pred = pred[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.6)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                
                if(label in choice_labels):
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    box_data = xyxy
                    box_data =[ int((int(box_data[0].item())/255)), int(int(box_data[0].item())%255), int((int(box_data[1].item())/255)), int(int(box_data[1].item()%255)), \
                        int((int(box_data[2].item())/255)), int(int(box_data[2].item()%255)), int((int(box_data[3].item())/255)), int(int(box_data[3].item()%255)) ]
                    box += box_data
    return box,im0


HOST = ""
PORT = 6001
ADDRESS = (HOST,PORT)
tcpServer = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcpServer.bind(ADDRESS)
tcpServer.listen(5)

while True:
    print("wait for connect\n")
    client_socket,client_address = tcpServer.accept()
    print("connect success\n")
    while True:
        cnt = 0
        img_bytes = b''
        start_time =time.perf_counter()
        while cnt < 640*480*3:
            data = client_socket.recv(1024)
            img_bytes += data
            cnt += len(data)
        if len(img_bytes) != 640*480*3:
            continue
        img = np.asarray(bytearray(img_bytes), dtype="uint8")
        img = img.reshape((480, 640, 3))
        # print("client to server time:{}ms".format((time.perf_counter() - start_time) * 1000))


        #图片 处理 开始
        start_time = time.perf_counter()
        box,img = detect(img)
        cv2.imshow('result', img)
        cv2.waitKey(1)
        # print("detect time:{}ms".format((time.perf_counter() - start_time) * 1000))
        #图片 处理 结束

        #图片 回传 开始
        start_time = time.perf_counter()
        client_socket.send(bytes(box))
        # print("time:{}ms".format((time.perf_counter()-start)*1000))
        # print("server to client time:{}ms".format((time.perf_counter() - start_time) * 1000))
        #图片 回传 结束
    cv2.destroyAllWindows()
    client_socket.close()
