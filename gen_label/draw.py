import cv2
from matplotlib import pyplot as plt
import os
'''
bboxs = open('./bbox.txt','r')
for line in bboxs.readlines():
    bbox = line.strip('/n').split(',')
    img = cv2.imread('./test1.jpg')
    x1 = int(float(bbox[0]))
    x2 = int(float(bbox[1]))
    y1 = int(float(bbox[2]))
    y2 = int(float(bbox[3]))
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 1)
    cv2.imwrite('./m.jpg',img)
    #cv2.imshow('im',img)
    #cv2.waitKey(0)
bboxs.close()

'''

#image_path = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test6.jpg'
image_path = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test/test6.jpg'
save_path = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test_pic/6'
bboxs = open('/lfs1/users/szhu/project/MTCNN-keras/pnet_out.txt','r')
img = cv2.imread(image_path)
for line in bboxs.readlines():
    bbox = line.strip('\n').split(' ')
    x1 = int(float(bbox[0]))
    y1 = int(float(bbox[1]))
    x2 = int(float(bbox[2]))
    y2 = int(float(bbox[3]))
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 1)
cv2.imwrite(save_path + '-1.jpg',img)

bboxs = open('/lfs1/users/szhu/project/MTCNN-keras/pnet_bout.txt','r')
img = cv2.imread(image_path)
for line in bboxs.readlines():
    bbox = line.strip('\n').split(' ')
    x1 = int(float(bbox[0]))
    y1 = int(float(bbox[1]))
    x2 = int(float(bbox[2]))
    y2 = int(float(bbox[3]))
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 1)
cv2.imwrite(save_path + '-2.jpg',img)

bboxs = open('/lfs1/users/szhu/project/MTCNN-keras/rnet_out_nonms.txt','r')
img = cv2.imread(image_path)
for line in bboxs.readlines():
    bbox = line.strip('\n').split(' ')
    x1 = int(float(bbox[0]))
    y1 = int(float(bbox[1]))
    x2 = int(float(bbox[2]))
    y2 = int(float(bbox[3]))
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 1)
cv2.imwrite(save_path + '-3.jpg',img)
bboxs = open('/lfs1/users/szhu/project/MTCNN-keras/onet_out_no.txt','r')
img = cv2.imread(image_path)
for line in bboxs.readlines():
    bbox = line.strip('\n').split(' ')
    x1 = int(float(bbox[0]))
    y1 = int(float(bbox[1]))
    x2 = int(float(bbox[2]))
    y2 = int(float(bbox[3]))
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 1)
cv2.imwrite(save_path + '-3-no.jpg',img)

bboxs = open('/lfs1/users/szhu/project/MTCNN-keras/onet_out.txt','r')
img = cv2.imread(image_path)
for line in bboxs.readlines():
    bbox = line.strip('\n').split(' ')
    x1 = int(float(bbox[0]))
    y1 = int(float(bbox[1]))
    x2 = int(float(bbox[2]))
    y2 = int(float(bbox[3]))
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 2)
    
lms = open('/lfs1/users/szhu/project/MTCNN-keras/landmark_out.txt','r')
for line in lms.readlines():
    lm = line.strip().split(' ')
    m = range(0,len(lm),2)
    for i in m:
        x1 = int(float(lm[i]))
        y1 = int(float(lm[i+1]))
        cv2.rectangle(img,(x1,y1),(x1,y1),(0,255,0), 4)

cv2.imwrite(save_path + '-4.jpg',img)
'''
path = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test/'
path1 = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test_pic/'
files = os.listdir(path)
for f in files:
    f = f.strip().split('.')[0]
    img = cv2.imread(path + f+'.jpg')
    bboxs = open(path + f + '.txt','r')
    
    for line in bboxs.readlines():
        bbox = line.strip('\n').split(' ')
        x1 = int(float(bbox[0]))
        y1 = int(float(bbox[1]))
        x2 = int(float(bbox[2]))
        y2 = int(float(bbox[3]))
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 1)
    cv2.imwrite(path1 + f + '_o.jpg',img)
'''