import cv2
import csv
import datetime
from datetime import date
from datetime import timedelta
thres = 0.70 # Threshold to detect object

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,70)
# print(cap)
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip(' ').split('\n')
# print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
k = input("Enter s to start\n")
if(k == 's'):
    i = 1
    flag = False
    while(True):
        success,img = cap.read()
        # cv2.imshow('',img)
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
            # print(classIds,bbox)

        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                prev = 'person'
                if(classNames[classId-1]!=prev and (classNames[classId-1]=='banana' or classNames[classId-1]=='apple' or classNames[classId-1]=='orange')):
                    print("added",classNames[classId-1])
                    prev = classNames[classId-1]
                    with open('items_list.csv', mode='a') as items_list:
                        items_writer = csv.writer(items_list, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        today = date.today()
                        date = datetime.datetime().strftime("%d-%m-%Y, %H:%M")
                        expiry = today + timedelta(days=3)
                        items_writer.writerow([i,classNames[classId-1], date, expiry])
                    flag = True
                    break
                cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        if(flag == True):
            break
        cv2.imshow('Output',img)
        cv2.waitKey(1)
