#coding:utf-8
import sys  
sys.path.append(r'./net')
sys.path.append(r'./gen_label') 
from PNet_p import PNet,PNet_cla,PNet_cla1
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

from callback import LossHistory,TBloss

import tensorflow as tf
from keras.backend.tensorflow_backend import  set_session
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.6
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

def process_line(line):
    tmp = line.strip().split(' ')
    img = load_img(tmp[0]) 
    x = img_to_array(img)
    x = x.astype('float32')
    x1 = image_generalize(x)
    cla = int(tmp[1])
    sample_weight_cla=1
    sample_weight_bbox=1
    if cla==-1:
        sample_weight_cla=0
    y1 = np.array([cla])
    bbox = map(float,tmp[2:])
    if bbox[0] == -1.0 and bbox[1] == -1.0:
        sample_weight_bbox = 0
    y2 =  np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

    return x, y1, y2,sample_weight_cla,sample_weight_bbox   

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
        X, Y, Y1,weight_cla,weight_bbox = [], [], [], [], []
        cnt = 0
        #sample_weight_cla = np.array([1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0])
        #sample_weight_bbox = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
        for i in range(len(lines)):
            x, y1,y2,sample_weight_cla,sample_weight_bbox= process_line(lines[i])
            X.append(x)
            Y.append(y1)
            Y1.append(y2)
            weight_cla.append(sample_weight_cla)
            weight_bbox.append(sample_weight_bbox)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                #print(len(Y))
                #print(len(weight_cla))
                #print(np.reshape(weight_cla,(len(weight_cla),)))
                yield(np.reshape(X,(len(X),12,12,3)),{'cla':np.reshape(Y,(len(Y),1,1,1)),
                      'bbox':np.reshape(Y1,(len(Y1),1,1,4))},{'cla':np.reshape(weight_cla,(len(weight_cla),)),
                        'bbox':np.reshape(weight_bbox,(len(weight_bbox),))})
                X, Y, Y1,weight_cla,weight_bbox = [], [], [], [], []
 
def claloss(y_true,y_pred):
    epsilon = 1e-4 
    print('y_true:',K.int_shape(y_true))

    print('mean(binary_crossentropy):',K.int_shape(K.mean(K.binary_crossentropy(y_pred,y_true[:,:,:,1:]),axis=-1)))
    print('sum(b_c) / sum(e+y_true):',K.int_shape(K.sum(y_true[:, :, :, :1]*K.binary_crossentropy(y_pred,y_true[:,:,:,1:]),axis=-1)/K.sum(epsilon + y_true[:, :, :, :1],axis=-1)))  
    print('sum(b_c,-1) / sum(e+y_true,-1):',K.int_shape(K.sum(y_true[:, :, :, :1]*K.binary_crossentropy(y_pred,y_true[:,:,:,1:]),axis=-1)/K.sum(epsilon + y_true[:, :, :, :1],axis=-1)))
    print('sum(e+y_true,-1):',K.int_shape(K.sum(epsilon + y_true[:, :, :, :1],axis=-1)))
    print('sum(b_c,k=true):',K.int_shape(K.sum(y_true[:, :, :, :1]*K.binary_crossentropy(y_pred,y_true[:,:,:,1:]),keepdims=True)))
    return  K.mean(K.binary_crossentropy(y_pred,y_true[:,:,:,:]),axis=-1)


def bboxloss(y_true,y_pred):
    epsilon = 1e-4
    return K.sum(y_true[:, :, :, :1] * K.square(y_pred[:,:,:,:] - y_true[:, :, :, 1:]))/ K.sum(epsilon + y_true[:, :, :, :1])
    


modelpath ='/lfs1/users/szhu/project/MTCNN-keras/model/model_rate/wrate-0.01-1-{epoch:02d}.h5'
log_filepath = '/lfs1/users/szhu/project/MTCNN-keras/loss/sgdrate'
batch_size = 40
num_classes = 3
nb_epoch = 10
bloss_path = './loss/file/batch_rate0.01_1_loss.txt'
eoss_path = './loss/file/epoch_rete0.01_1_loss.txt'
vloss_path = './loss/file/validation_rate0.01_1_acc.txt'

train_path = './data4/12/label-train.txt'
test_path = './data4/12/label-test.txt'
model = PNet_cla1()
sgd = optimizers.SGD(lr=0.01,momentum=0.9)
  #adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
tb_cb = TBloss(log_dir=log_filepath,batch_size=batch_size) 
model.summary()
model.compile(optimizer=sgd,
        loss={'cla':claloss,'bbox':mean_squared_error},
        loss_weights={'cla':1,'bbox':0.5},
        metrics=['accuracy'])
history = LossHistory(bloss_path,eoss_path,vloss_path)
model_per_epoch = ModelCheckpoint(modelpath, save_weights_only=True, period=1,verbose=1)
'''
model.fit_generator(generate_arrays_from_file(train_path,batch_size,num_class=num_classes),  
                        steps_per_epoch=300,nb_epoch=nb_epoch,
                        validation_data=generate_arrays_from_file(test_path,batch_size,num_class=num_classes),
                        validation_steps=20,callbacks=[history,model_per_epoch,tb_cb])
'''

model.fit_generator(generate_arrays_from_file(train_path,batch_size,num_class=num_classes),  
                        steps_per_epoch=305100,nb_epoch=nb_epoch,
                        validation_data=generate_arrays_from_file(test_path,batch_size,num_class=num_classes),
                        validation_steps=33900,callbacks=[history,model_per_epoch,tb_cb])




    