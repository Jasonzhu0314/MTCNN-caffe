#coding:utf-8
import sys  
sys.path.append(r'./net')
sys.path.append(r'./gen_label') 
from PNet_p import PNet
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
config.gpu_options.per_process_gpu_memory_fraction=0.7
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
    X, Y1, Y2 = [], [], []
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
        bbox = map(float,tmp[2:])
        if bbox[0] == -1.0 and bbox[1] == -1.0: # 负样本
            y2 = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]) 
        else:
            y2 = np.array([1, bbox[0], bbox[1], bbox[2], bbox[3]])
        X.append(x1)
        Y1.append(y1)
        Y2.append(y2)
    #print(X.shape)  
    X = np.reshape(X,(len(X),12,12,3))
    Y1 = np.reshape(Y1,(len(Y1),1,1,2))
    Y2 = np.reshape(Y2,(len(Y2),1,1,5))  
    return X, Y1 ,Y2  

def load_file(filename):
	line = []
	with open(filename,'r') as f:
		lines = f.readlines()
		for i in lines:
			line.append(i)
	#print(line)
	return line


def generate_arrays_from_file(path,batch_size,num_class=3,sample_num=200):
    lines = load_file(path)
    while 1:
        samples = random.sample(lines, sample_num)
        X, Y1, Y2 = process_line(samples)
        for idx in range(sample_num/batch_size):   
            yield (X[idx*batch_size:(idx+1)*batch_size],
                    {'cla':Y1[idx*batch_size:(idx+1)*batch_size],
                     'bbox':Y2[idx*batch_size:(idx+1)*batch_size]})
 
def claloss(y_true,y_pred):   
    epsilon1 = 1e-4
    return  K.sum(y_true[:, :, :, :1] * K.binary_crossentropy(y_pred[:,:,:,:],y_true[:,:,:,1:])) / K.sum(epsilon1 + y_true[:, :, :, :1])


def bboxloss(y_true,y_pred):
    epsilon = 1e-4
    return K.sum(y_true[:, :, :, :1] * K.square(y_pred[:,:,:,:] - y_true[:, :, :, 1:]))/ K.sum(epsilon + y_true[:, :, :, :1])
    
def new_binary_accuracy(y_true,y_pred):
    #return K.mean(K.equal(y_true[:,:,:,1:], K.round(y_pred)), axis=-1)
    return K.mean(K.equal(y_true[:,:,:,1:], K.round(y_pred)), axis=-1)

def new_bbox_accuracy(y_true, y_pred):
    #return K.mean(K.abs(y_pred - y_true[:,:,:,1:]), axis=-1)
    epsilon = 1e-4
    return K.sum(y_true[:, :, :, :1] * K.abs(y_pred[:,:,:,:] - y_true[:, :, :, 1:]), axis=-1)/ K.sum(epsilon + y_true[:, :, :, :1])

modelpath ='/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/w0.001-1-{epoch:02d}.h5'
log_filepath = '/lfs1/users/szhu/project/MTCNN-keras/loss/sgd2'
batch_size = 80
num_classes = 3
nb_epoch = 10
train_path = './data5/12/label-train.txt'
test_path = './data5/12/label-test.txt'
model = PNet()
model.load_weights('/lfs1/users/szhu/project/MTCNN-keras/model/real_accuracy/w0.01-1-02.h5')
sgd = optimizers.SGD(lr=0.001,momentum=0.9)
  #adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
tb_cb = TBloss(log_dir=log_filepath,batch_size=batch_size) 
model.summary()
model.compile(optimizer=sgd,
      loss={'cla':claloss,'bbox':bboxloss},
        loss_weights={'cla':1.0,'bbox':0.5},
        metrics=[new_binary_accuracy,new_bbox_accuracy])
#history = LossHistory()
model_per_epoch = ModelCheckpoint(modelpath, save_weights_only=True, period=1,verbose=1)
'''
model.fit_generator(generate_arrays_from_file(train_path,batch_size,num_class=num_classes),  
                        steps_per_epoch=300,nb_epoch=nb_epoch,
                        validation_data=generate_arrays_from_file(test_path,batch_size,num_class=num_classes),
                        validation_steps=20,callbacks=[history,model_per_epoch,tb_cb])
'''
model.fit_generator(generate_arrays_from_file(train_path,batch_size,num_class=num_classes),  
                        steps_per_epoch=152550,nb_epoch=nb_epoch,
                        validation_data=generate_arrays_from_file(test_path,batch_size,num_class=num_classes),
                        validation_steps=16950,callbacks=[model_per_epoch,tb_cb])




    