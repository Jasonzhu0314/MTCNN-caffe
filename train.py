#coding:utf-8
from PNet import PNet
from keras import optimizers,metrics,utils
from keras.losses import categorical_crossentropy,mean_squared_error
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
import numpy.random
import scipy.ndimage
import keras.backend as K
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers,metrics
from keras.backend.tensorflow_backend import  set_session
from utils import convert_cla,convert_bbox

from callback import LossHistory
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
set_session(tf.Session(config=config))

def image_generalize(image): 
      image[:,:,0] = image[:,:,0] - 127.5
      image[:,:,1] = image[:,:,1] - 127.5
      image[:,:,2] = image[:,:,2] - 127.5
      #image_crop = image_crop[:, :, ::-1]
      #image.astype(numpy.float32)
      image *=0.0078125
      #print image_crop
      #image_crop = misc.imresize(image_crop, image.shape)
      return image

def process_line(line):  
    tmp = line.strip().split(' ')
    img = load_img(tmp[0]) 
    x = img_to_array(img)
    x = x.astype('float32')
    x1 = image_generalize(x)    
    y1 = numpy.array(int(tmp[1]))
    y2 = numpy.array(map(float,tmp[2:]))  
    return x1, y1 ,y2  

def generate_arrays_from_file(path,batch_size,num_class=2):  
    while 1:  
        f = open(path)  
        cnt = 0  
        X = []
        Y1 = []  
        Y2 = []
        for line in f:  
            # create Numpy arrays of input data  
            # and labels, from each line in the file  
            x, y1, y2 = process_line(line)
            X.append(x)
            Y1.append(y1)
            Y2.append(y2)
            #print np.array(X).shape,np.array(Y1).shape,np.array(Y2).shape
            cnt += 1  
            if cnt==batch_size:  
                cnt = 0 
                yield (np.array(X),{'cla':convert_cla(np.array(Y1), num_class),
                	                   'bbox':convert_bbox(np.array(Y2),4)})
                #  两个任务，加入两个标签
                X = []  
                Y1 = [] 
                Y2 = [] 
    f.close()  


def train():
	batch_size = 5
	num_classes = 2
	nb_epoch = 20
	train_path = './data/12/label-train.txt'
	model = PNet()
	sgd = optimizers.SGD(lr=0.001,momentum=0.9)
	model.summary()
	model.compile(optimizer=sgd,
			loss={'cla':categorical_crossentropy,'bbox':mean_squared_error},
				loss_weights={'cla':1,'bbox':0.5},
				metrics=['accuracy'])
	model.fit_generator(generate_arrays_from_file(train_path,batch_size,num_class=num_classes),  
                        steps_per_epoch=757,nb_epoch=nb_epoch)
if __name__ == '__main__':
	train()
	#label_dimension()


    