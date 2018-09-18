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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
set_session(tf.Session(config=config))




stdsize1 = 241
stdsize = 48
anno_file = "../label/wider_gt.txt"
im_dir = "samples"

image_dir = '/lfs1/users/szhu/project/MTCNN-keras/data4/'
save_dir = "/lfs1/users/szhu/project/MTCNN-keras/data4/" + str(stdsize)
pos_save_dir = image_dir+str(stdsize) + "/positive"
part_save_dir = image_dir+str(stdsize) + "/part"
neg_save_dir = image_dir+str(stdsize) + '/negative'

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
print "%d pics in total" % num # 计算annotation中的所有的图片。
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


def detect_face(img,gts,imgpath,p_model,r_model):
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
        
    numbox = total_boxes.shape
    #print numbox
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
    print(total_boxes.shape)
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
        #print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                print("[6]:",total_boxes.shape[0])
                #total_boxes = bbreg(total_boxes, mv[:, pick])
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

# 初始化PNet
p_model = PNet_p.PNet()
p_model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/w0.001-1-02.h5')
r_model= PNet.RNet()
r_model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/r0.001-1-09.h5')
annotation = annotations[0]
annotation = annotation.strip().split(' ')
gt = map(float, annotation[1:])
gts = np.array(gt, dtype=np.float32).reshape(-1, 4) # 转化为（1,4）数组
gts = np.array([194.,89.,333.,263.]).reshape(-1, 4)
image_path = annotation[0]
image_path = '/lfs1/users/szhu/project/MTCNN-keras/gen_label/test2.jpg'
#img = io.imread(image_path) 
x= load_img(image_path)
img = img_to_array(x)
img2 = img
#img = img.astype('float32')
h = img.shape[0]
w = img.shape[1]
detect_face(img,gts,image_path,p_model,r_model)
'''
predictions = p_model.predict(img)
cla = predictions[0]
bbox = predictions[1]
print(cla.shape)
print(bbox.shape)
'''
    #detect_face(img,boxes,annotation[0])










'''
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print idx, "images done"

    height, width, channel = img.shape

    neg_num = 0
    while neg_num < 100:
        size = npr.randint(40, min(width, height) / 2) # 计算原图片大小的二分之一，在40和图片大小二分之一之间生成随机整数
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        # 在图片中随机crop图片大小40到图片大小的二分之一之内
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)
        # crop的图片resize到（12,12）
        if np.max(Iou) < 0.3:
            # Iou低于0.3的为负样本，保存下来
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(str(stdsize1)+"/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    
    for box in boxes:
        
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1  # 计算bbox的人脸的长宽，抛弃一些比较小的人脸
    
        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 12 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
            continue
        #print('generate_positive')
        # generate positive examples and part faces
        for i in range(50):
            try:
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            # ceil向正方向取整，随机初始化（bbox最小长度的0.8倍，1.25倍的长和宽中的最大值）
            # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height: # crop坐标超出图片大小，跳入下一个
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx1) / float(size)
                offset_y2 = (y2 - ny1) / float(size)
            # 计算四个bbox的坐标偏差
                cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write(str(stdsize1)+"/positive/%s"%p_idx + ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write(str(stdsize1)+"/part/%s"%d_idx + ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            except:
                continue
        box_idx += 1
        print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)
    
f1.close()
f2.close()
f3.close()
'''