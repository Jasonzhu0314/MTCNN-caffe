#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU,IoU2
sys.path.append('../')
from net import PNet_p,PNet
from skimage import transform,io
from keras.preprocessing.image import load_img,img_to_array

import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import  set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.1
set_session(tf.Session(config=config))


stdsize1 = 241
stdsize = 48



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

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print("bb", boundingbox)
    return boundingbox


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

    #print("dy"  ,dy )
    #print("dx"  ,dx )
    #print("y "  ,y )
    #print("x "  ,x )
    #print("edy" ,edy)
    #print("edx" ,edx)
    #print("ey"  ,ey )
    #print("ex"  ,ex )


    #print('boxes', boxes)
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
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
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
    
    #print('bboxA', bboxA)
    #print('w', w)
    #print('h', h)
    #print('l', l)
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA   # 扩展整个图片的大小，按最大值扩大

def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    dx1 = reg[:,:,0]
    dy1 = reg[:,:,1]
    dx2 = reg[:,:,2]
    dy2 = reg[:,:,3]
    (x, y) = np.where(map >= t)  # 将score>=阈值的坐标找出来
    # x是一个array一维的数组，y也是一个array一维的数组，(array([2,2,2],array([1,2,3])))
    yy = y
    xx = x
    score = map[x,y]
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


def detect_face(img,gts,imgpath,p_model,r_model,o_model):
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
        img_x = cv2.resize(img,(hs,ws),interpolation=cv2.INTER_LINEAR)

        img_data = image_generalize(img_x)
        
        img_data = np.expand_dims(img_data,axis=0)
        img_data1 = img_data
        out = p_model.predict(img_data)
        cla = out[0][0,:,:,0]
        bbox = out[1][0]
        #print(cla.shape)
        boxes = generateBoundingBox(cla,bbox,scale,0.6)
        if boxes.shape[0] != 0:
            #print(boxes[4:9])
            #print('im_data', im_data[0:5, 0:5, 0], '\n')
            #print('prob1', out['prob1'][0,0,0:3,0:3])

            pick = nms(boxes, 0.5, 'Union')
            #print('pick:',pick)
            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
        
    numbox = total_boxes.shape[0]
    print('Pnet:{}'.format(numbox))
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        if len(pick) > 0 :
        	total_boxes = total_boxes[pick, :]
        	t1 = total_boxes[:,0]
        	t2 = total_boxes[:,1]
        	t3 = total_boxes[:,2]
        	t4 = total_boxes[:,3]
        
        	p = open('/lfs1/users/szhu/project/MTCNN-keras/pnet_out.txt','w')
        	#p.write(imgpath)
        	#p.write(' ')
        	for i in range(t1.shape[0]):
           		p.write(str(t1[i]))
           		p.write(' ')
           		p.write(str(t2[i]))
           		p.write(' ')
           		p.write(str(t3[i]))
           		p.write(' ')
           		p.write(str(t4[i]))
           		p.write('\n')
        	p.close()
        	# revise and convert to square
        	regh = total_boxes[:,3] - total_boxes[:,1]  
        	regw = total_boxes[:,2] - total_boxes[:,0]
        	t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        	t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        	t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        	t4 = total_boxes[:,3] + total_boxes[:,8]*regh
     		p = open('/lfs1/users/szhu/project/MTCNN-keras/pnet_bout.txt','w')
        	#p.write(imgpath)
        	#p.write(' ')
        	for i in range(t1.shape[0]):
           		p.write(str(t1[i]))
           		p.write(' ')
           		p.write(str(t2[i]))
           		p.write(' ')
           		p.write(str(t3[i]))
           		p.write(' ')
           		p.write(str(t4[i]))
           		p.write('\n')
        	p.close()
        	total_boxes = np.array([t1,t2,t3,t4]).T
        	total_boxes = rerec(total_boxes)
        	#print("[4]:",total_boxes.shape[0])
        	total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        	#print("[4.5]:",total_boxes.shape[0])
        
        	[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
        	#print( tmpw[11], tmph[11])
    numbox = total_boxes.shape[0]
    print('Pnet+nms:{}'.format(numbox))
    if numbox > 0:
        # second stage

        #print('tmph', tmph)
        #print('tmpw', tmpw)
        #print("y,ey,x,ex", y, ey, x, ex, )
        #print("edy", edy)

        #tempimg = np.load('tempimg.npy')

        
        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            if tmph[k]<0 or tmpw[k]<0 or int(x[k])==int(ex[k])+1 or int(y[k]) == int(ey[k])+1:
                continue
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            #print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            #print("tmp", tmp.shape)
            
            tempimg[k,:,:,:] =cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
        #tempimg = (resized_img-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet
        print(tempimg.shape)
        img_data = (tempimg-127.5)*0.0078125

        
        out = r_model.predict(img_data)
        score = out[0]
        bbox = out[1]
        print(out[0].shape)
        print(out[1].shape)
        print(total_boxes.shape)
        #print('score', score)
        pass_t = np.where(score>0.7)[0]
        print('pass_t', pass_t)
        
        score =  np.array(score[pass_t])
        print(score.shape)
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        print("[5]:",total_boxes.shape[0])
        #print(total_boxes)

        #print("1.5:",total_boxes.shape)
        
        mv = out[1][pass_t, :].T
        numbox = total_boxes.shape[0]
        print('Rnet:{}'.format(numbox))
        #print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                p = open('/lfs1/users/szhu/project/MTCNN-keras/rnet_out_nonms.txt','w')
                t1 = total_boxes[:,0]
                t2 = total_boxes[:,1]
                t3 = total_boxes[:,2]
                t4 = total_boxes[:,3]
                for i in range(t1.shape[0]):
                    p.write(str(t1[i]))
                    p.write(' ')
                    p.write(str(t2[i]))
                    p.write(' ')
                    p.write(str(t3[i]))
                    p.write(' ')
                    p.write(str(t4[i]))
                    p.write('\n')
                p.close()
                print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                print("[8]:",total_boxes.shape[0])
        numbox = total_boxes.shape[0]
        if numbox > 0:
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                if tmph[k]<0 or tmpw[k]<0 or int(x[k])==int(ex[k])+1 or int(y[k]) == int(ey[k])+1:
                    continue
                tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] =cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
            img_data = (tempimg-127.5)*0.0078125

        
            out = o_model.predict(img_data)
            score = out[0]
            bbox = out[1]
            print(out[0].shape)
            print(out[1].shape)
            print(total_boxes.shape)
            #print('score', score)
            pass_t = np.where(score>0.7)[0]
            print('pass_t', pass_t)
            
            score =  np.array(score[pass_t])
            print(score.shape)
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
            print("[5]:",total_boxes.shape[0])
            #print(total_boxes)

            #print("1.5:",total_boxes.shape)
            mv = out[1][pass_t, :].T
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.6, 'Min')
                
                #print(pick)
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    print("[11]:",total_boxes.shape[0])
                    txt_path = imgpath.strip('jpg')+'txt'
                    p = open(txt_path,'w')
                    t1 = total_boxes[:,0]
                    t2 = total_boxes[:,1]
                    t3 = total_boxes[:,2]
                    t4 = total_boxes[:,3]
                    for i in range(t1.shape[0]):
                        p.write(str(t1[i]))
                        p.write(' ')
                        p.write(str(t2[i]))
                        p.write(' ')
                        p.write(str(t3[i]))
                        p.write(' ')
                        p.write(str(t4[i]))
                        p.write('\n')
                    p.close()
    return total_boxes    

import os
# 初始化PNet
p_model = PNet_p.PNet()
p_model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/w0.001-1-02.h5')
r_model= PNet.RNet()
r_model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/r0.01-6-14.h5')
o_model = PNet.ONet()
o_model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/o0.001-1-19.h5')

path = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test/'
files = os.listdir(path)
for f in files:
	gts = np.array([194.,89.,333.,263.]).reshape(-1, 4)
	image_path = path + f
	#img = io.imread(image_path) 
	x= load_img(image_path)
	img = img_to_array(x)
	img2 = img
	#img = img.astype('float32')
	h = img.shape[0]
	w = img.shape[1]
	detect_face(img,gts,image_path,p_model,r_model,o_model)
