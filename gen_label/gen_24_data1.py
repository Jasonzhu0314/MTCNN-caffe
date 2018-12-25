#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU,IoU2
sys.path.append('../')
from net import Net
from skimage import transform,io
from keras.preprocessing.image import load_img,img_to_array

import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import  set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
set_session(tf.Session(config=config))

stdsize = 24
anno_file = "../label/wider_gt.txt"
im_dir = "samples"

image_dir = '/home/users/zhuzhengshuai/data/mtcnn/data/'
save_dir = image_dir + str(stdsize)
pos_save_dir = save_dir + "/positive"
part_save_dir = save_dir + "/part"
neg_save_dir = save_dir + '/negative'

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')
#f4 = open(os.path.join(save_dir, 'r_net_label.txt'),'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num) # 计算annotation中的所有的图片。
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
total_idx = 0



def image_generalize(image):
    image[:,:,0] = image[:,:,0] - 127.5
    image[:,:,1] = image[:,:,1] - 127.5
    image[:,:,2] = image[:,:,2] - 127.5
    image *=0.0078125
    return image

def process_line(annotation):  
    annotation = annotation.strip().split(' ')
    bbox = map(float, annotation[1:]) 
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4) # 转化为（1,4）数组
    img = load_img(annotation[0]) 
    x = img_to_array(img)
    x = x.astype('float32')
    x = image_generalize(x)
    return x,boxes

def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print('#################')
    #print('boxes', boxes)
    #print('w,h', w, h)
    
    tmph = boxes[:,3] - boxes[:,1] + 1  # 预测出来的框的长度和宽度
    tmpw = boxes[:,2] - boxes[:,0] + 1  #  
    numbox = boxes.shape[0]             # bbox的数量


    dx = np.ones(numbox)                # 49个1
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph                          # edx = 宽度

    x = boxes[:,0:1][:,0]        # x1坐标
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]       # y2坐标
   
   
    tmp = np.where(ex > w)[0]   # 如果x2坐标大于原始图片宽度大小w
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]    # 预测出来的宽度edx和x2都要改变，将x2=w-1
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]     # y2=h-1
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])     # x <1,x=0

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)

    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]

def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)  # ç›¸ä¹˜
    I = np.array(s.argsort()) # 从小到大排序，返回下标字母
    
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        #print(o)
        pick.append(I[-1])
        I = I[np.where( o < threshold)[0]]
    return pick

def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    

    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA   # 扩展整个图片的大小，按最大值扩大

def generateBoundingBox(cla, reg, scale, t):
    stride = 2
    cellsize = 12
    dx1 = reg[:,:,0]
    dy1 = reg[:,:,1]
    dx2 = reg[:,:,2]
    dy2 = reg[:,:,3]
    (x, y) = np.where(cla >= t)  # 将score>=阈值的坐标找出来
    # x是一个array一维的数组，y也是一个array一维的数组，(array([2,2,2],array([1,2,3])))
    yy = y
    xx = x
    score = cla[x,y]
    #print(x,y)
    #print score
    #print(x,y)
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])
    # 回归的坐标的取值
    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T  # å°†[yy,xx]çš„å¤§å°w*2,è½¬åŒ–ä¸?*w
    #print(boundingbox)
    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print('(x,y)',x,y)
    #print('score', score)
    #print('reg', reg)

    return boundingbox_out.T


def detect_face(img,gts,imgpath,p_model):
    image_copy = cv2.imread(imgpath)
    

    factor = 0.709
    minsize = 20
    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)

    h = img.shape[0]
    w = img.shape[1]


    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m  # 0.6倍的宽一直到12

    # create scale pyramid
    scales = [0.6]
    
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))
        img_x = transform.resize(img,(hs,ws))

        img_data = image_generalize(img_x)
        
        img_data = np.expand_dims(img_data,axis=0)
        img_data1 = img_data
        out = p_model.predict(img_data)
        cla = out[0][0,:,:,0]
        bbox = out[1][0]
        #print(cla.shape)
        boxes = generateBoundingBox(cla,bbox,scale,0.6)
        if boxes.shape[0] != 0:

            pick = nms(boxes, 0.5, 'Union')
            #print('pick:',pick)
            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
        
    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        if len(pick) > 0 :
        	total_boxes = total_boxes[pick, :]
        
        	# revise and convert to square
        	regh = total_boxes[:,3] - total_boxes[:,1]  
        	regw = total_boxes[:,2] - total_boxes[:,0]
        	t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        	t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        	t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        	t4 = total_boxes[:,3] + total_boxes[:,8]*regh
     
        	total_boxes = np.array([t1,t2,t3,t4]).T
        	total_boxes = rerec(total_boxes)
        	#print("[4]:",total_boxes.shape[0])
        	total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        	#print("[4.5]:",total_boxes.shape[0])
        
        	[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
        	#print( tmpw[11], tmph[11])
    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            
            if tmph[k]<0 or tmpw[k]<0 or int(x[k])==int(ex[k])+1 or int(y[k]) == int(ey[k])+1:
                continue
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
            try:

                    tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = image_copy[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            
                    resized_img =cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
                    crop = [int(x[k]),int(y[k]),int(ex[k]+1),int(ey[k]+1)]
                    size = int(ey[k])+1-int(y[k])+1
                    Iou,index = IoU2(crop,gts)
                    if(Iou >= 0.65):
                        global p_idx
                        offset_x1 = (gts[index][0] - crop[0])/float(size) 
                        offset_y1 = (gts[index][1] - crop[1])/float(size)
                        offset_x2 = (gts[index][2] - crop[2])/float(size) 
                        offset_y2 = (gts[index][3] - crop[3])/float(size)
                        save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                        f1.write(pos_save_dir+'/'+"%s"%p_idx + '.jpg'+ ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_img)
                
                        p_idx += 1
                    elif(Iou >= 0.4):
                        global d_idx
                        offset_x1 = (gts[index][0] - crop[0])/float(size) 
                        offset_y1 = (gts[index][1] - crop[1])/float(size)
                        offset_x2 = (gts[index][2] - crop[2])/float(size) 
                        offset_y2 = (gts[index][3] - crop[3])/float(size)
                        save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                        f3.write(part_save_dir+'/'+"%s"%d_idx + '.jpg'+ ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                        cv2.imwrite(save_file, resized_img)
                        d_idx += 1
                    elif(Iou < 0.3):
                        global n_idx
                        save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                        f2.write(neg_save_dir+'/'+"%s"%n_idx + '.jpg' + ' 0 -1 -1 -1 -1\n')
                        cv2.imwrite(save_file, resized_img)
                
                        n_idx += 1
                    
                    
            except:
                    continue
    global total_idx
    total_idx += 1
    if total_idx % 50 == 0:
        print('total:',total_idx)
        print('positive:',p_idx)
        print('negative:',n_idx)
        print('part:',d_idx)


# 初始化PNet
p_model = Net.PNet()
p_model.load_weights('../model/sgd/weights-improv-961-0.75.h5')
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    gt = list(map(float, annotation[1:]))
    gts = np.array(gt, dtype=np.float32).reshape(-1, 4) # 转化为（1,4）数组
    image_path = annotation[0]
    x= load_img(image_path)
    img = img_to_array(x)
    img2 = img
    detect_face(img,gts,image_path,p_model)
