#coding:utf-8
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU,IoU1



stdsize = 12
anno_file = "../label/wider_gt.txt"
im_dir = "samples"

image_dir = "/home/users/zhuzhengshuai/data/mtcnn/data/"
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
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)  # 计算annotation中的所有的图片。
p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # dont care
idx = 0
box_idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    # print(annotation)
    bbox = list(map(float, annotation[1:]))
    boxes = np.array(bbox).reshape(-1, 4)  # 转化为（1,4）数组
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")

    height, width, channel = img.shape
    
    neg_num = 0
    while neg_num < 300:
        size = npr.randint(12, min(width, height) / 2)  # 计算原图片大小的二分之一，在40和图片大小二分之一之间生成随机整数
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        # 在图片中随机crop图片大小40到图片大小的二分之一之内
        Iou = IoU1(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)
        # crop的图片resize到（12,12）
        if Iou < 0.3:
            # Iou with all gts must below 0.3
            # Iou低于0.3的为负样本，保存下来
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(str(stdsize)+"/negative/%s" % n_idx + ' 0\n')
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
        
         # generate negative examples that have overlap with gt
        for i in range(60):
            size = npr.randint(12,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[int(ny1) : int(ny1 + size), int(nx1) : int(nx1 + size), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                f2.write("12/negative/%s"%n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
        
        # generate positive examples and part faces
        for i in range(200):
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
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
            # 计算四个bbox的坐标偏差
                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    f1.write(str(stdsize)+"/positive/%s"%p_idx + ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                    f3.write(str(stdsize)+"/part/%s"%d_idx + ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            except:
                continue
        box_idx += 1
        print("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
    
f1.close()
f2.close()
f3.close()
