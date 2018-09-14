#coding:utf-8
import sys  
sys.path.append(r'./net')
sys.path.append(r'./gen_label') 
from PNet import ONet
from keras import optimizers,metrics,utils
from keras.losses import categorical_crossentropy,mean_squared_error
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from keras.callbacks import LearningRateScheduler, ModelCheckpoint ,TensorBoard
import numpy as np
import random
import scipy.ndimage

import os
import matplotlib.pyplot as plt

from keras import optimizers,metrics

from utils import convert_cla,convert_bbox

from callback1 import LossHistory,TBloss

import tensorflow as tf
from keras.backend.tensorflow_backend import  set_session
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
set_session(tf.Session(config=config))

lambda_cls = 1
lambda_bbox = 0.5

def image_generalize(image): 
      image[:,:,0] = image[:,:,0] - 127.5
      image[:,:,1] = image[:,:,1] - 127.5
      image[:,:,2] = image[:,:,2] - 127.5
      #image_crop = image_crop[:, :, ::-1]
      #image.astype(numpy.float32)
      image *=0.0078125
      return image

def process_line(lines):
    X, Y1, Y2, Y3 = [], [], [], []
    for line in lines:  
        tmp = line.strip().split(' ')
        img = load_img(tmp[0]) 
        x = img_to_array(img)
        x = x.astype('float32')
        x1 = image_generalize(x)
        cla = int(tmp[1])
        if cla == 0:  # 负样本
            y1 = np.array([1, 0])
        elif cla == 1:  #正样本
            y1 = np.array([1, 1])
        else:  # 部分人脸
            y1 = np.array([0, -1])
        bbox = map(float,tmp[2:6])
        if bbox[0] == -1.0 and bbox[1] == -1.0: # 负样本
            y2 = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]) 
        else:
            y2 = np.array([1, bbox[0], bbox[1], bbox[2], bbox[3]])
        l = map(float,tmp[6:])
        if l[0] == 0. and l[3] == 0.:

            y3 = np.array([0,l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9]])
        else:
            y3 = np.array([1,l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9]])
        #print(y3)
        X.append(x1)
        Y1.append(y1)
        Y2.append(y2)
        Y3.append(y3)
    #print(X.shape)  
    X = np.reshape(X,(len(X),48,48,3))
    Y1 = np.reshape(Y1,(len(Y1),2))
    Y2 = np.reshape(Y2,(len(Y2),5))
    Y3 = np.reshape(Y3,(len(Y3),11))  
    return X, Y1, Y2 ,Y3 

def load_file(filename):
    line = []
    with open(filename,'r') as f:
        lines = f.readlines()
        for i in lines:
            line.append(i)
    #print(line)
    return line


def generate_arrays_from_file(path,batch_size,sample_num=200):
    lines = load_file(path)
    while 1:
        samples = random.sample(lines, sample_num)
        X, Y1, Y2, Y3 = process_line(samples)
        for idx in range(sample_num/batch_size):   
            yield (X[idx*batch_size:(idx+1)*batch_size],
                    {'cla':Y1[idx*batch_size:(idx+1)*batch_size],
                     'bbox':Y2[idx*batch_size:(idx+1)*batch_size],
                     'landmark':Y3[idx*batch_size:(idx+1)*batch_size]})
 
def claloss(y_true,y_pred):   
    epsilon1 = 1e-4
    return  K.sum(y_true[:, :1] * K.binary_crossentropy(y_pred[:,:],y_true[:,1:])) / K.sum(epsilon1 + y_true[:, :1])


def bboxloss(y_true,y_pred):
    epsilon = 1e-4
    return K.sum(y_true[:, :1] * K.square(y_pred[:,:] - y_true[:, 1:]))/ K.sum(epsilon + y_true[:, :1])
    
def landmarkloss(y_true,y_pred):
    epsilon = 1e-4
    return K.sum(y_true[:, :1] * K.square(y_pred[:,:] - y_true[:, 1:]))/ K.sum(epsilon + y_true[:, :1])
def new_binary_accuracy(y_true,y_pred):
    #return K.mean(K.equal(y_true[:,:,:,1:], K.round(y_pred)), axis=-1)
    return K.mean(K.equal(y_true[:,1:], K.round(y_pred)), axis=-1)

def new_bbox_accuracy(y_true, y_pred):
    #return K.mean(K.abs(y_pred - y_true[:,:,:,1:]), axis=-1)
    epsilon = 1e-4
    return K.sum(y_true[:, :1] * K.abs(y_pred[:,:] - y_true[:, 1:]), axis=-1)/ K.sum(epsilon + y_true[:, :1])

def new_landmark_accuracy(y_true, y_pred):
    epsilon = 1e-4
    return K.sum(y_true[:, :1] * K.abs(y_pred[:,:] - y_true[:, 1:]), axis=-1)/ K.sum(epsilon + y_true[:, :1])

base_lr = 0.001
modelpath ='/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/o'+str(base_lr)+'-0-{epoch:02d}.h5'
log_filepath = '/lfs1/users/szhu/project/MTCNN-keras/loss/sgdo'+str(base_lr)+'-0'
batch_size = 80
nb_epoch = 20
num_classes = 3
train_path = './data4/48/label-train1.txt'
test_path = './data4/48/label-test1.txt'
model = ONet()
model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/o0.01-1-12.h5')
sgd = optimizers.SGD(lr=base_lr,momentum=0.9)
  #adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
tb_cb = TBloss(log_dir=log_filepath,batch_size=batch_size) 
model.summary()
model.compile(optimizer=sgd,
      loss={'cla':claloss,'bbox':bboxloss,'landmark':landmarkloss},
        loss_weights={'cla':1.0,'bbox':0.5,'landmark':1.0},
        metrics=[new_binary_accuracy,new_bbox_accuracy,new_landmark_accuracy])
#history = LossHistory()
model_per_epoch = ModelCheckpoint(modelpath, save_weights_only=True, period=1,verbose=1)
'''
model.fit_generator(generate_arrays_from_file(train_path,batch_size,num_class=num_classes),  
                        steps_per_epoch=300,nb_epoch=nb_epoch,
                        validation_data=generate_arrays_from_file(test_path,batch_size,num_class=num_classes),
                        validation_steps=20,callbacks=[history,model_per_epoch,tb_cb])
'''
model.fit_generator(generate_arrays_from_file(train_path,batch_size),  
                        steps_per_epoch=6562,nb_epoch=nb_epoch,
                        validation_data=generate_arrays_from_file(test_path,batch_size),
                        validation_steps=730,callbacks=[model_per_epoch,tb_cb])




    