from keras.models import Model
from keras.layers import Dense, Activation, Input, Conv2D, MaxPooling2D, Flatten, Dropout,Input
from keras.layers.advanced_activations import PReLU
from keras.utils import plot_model
from keras.models import load_model
from keras import optimizers,initializers
from keras.activations import softmax,sigmoid
import h5py

def PNet():
    image_input = Input(shape=(None,None,3))
    xavier = initializers.glorot_uniform()

    x = Conv2D(10,(3,3), kernel_initializer=xavier, name='conv1')(image_input)
    x = PReLU(name='prelu1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(16,(3,3), kernel_initializer=xavier, name='conv2')(x)
    x = PReLU(name='prelu2')(x)

    x = Conv2D(32,(3,3), kernel_initializer=xavier, name='conv3')(x)
    x = PReLU(name='prelu3')(x)

    cla = Conv2D(2,(1,1), kernel_initializer=xavier, name='cla',activation=softmax)(x)
    
    bbox = Conv2D(4,(1,1), kernel_initializer=xavier, name='bbox')(x)

    model = Model(inputs=image_input,outputs=[cla,bbox])
    
    return model

def RNet():
    image_input = Input(shape=(24,24,3))
    xavier = initializers.glorot_uniform()
    x = Conv2D(28,(3,3),kernel_initializer=xavier, name='conv1')(image_input)
    x = PReLU(name='prelu1')(x)
    x = MaxPooling2D((3,3),strides=(2,2), name='pool1')(x)

    x = Conv2D(48,(3,3),kernel_initializer=xavier, name='conv2')(x)
    x = PReLU(name='prelu2')(x)
    x = MaxPooling2D((3,3),strides=(2,2), name='pool2')(x)
   
    x = Conv2D(64,(2,2),kernel_initializer=xavier, name='conv3')(x)
    x = PReLU(name='prelu3')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(128)(x)
    x = PReLU(name='prelu4')(x)

    cla = Dense(1,activation=sigmoid,kernel_initializer=xavier,name='cla')(x)
    bbox = Dense(4,kernel_initializer=xavier,name='bbox')(x)

    model = Model(inputs=image_input,outputs=[cla,bbox])

    return model
    #loss_weight:cla=1,bbox=0.5

def ONet():
    image_input = Input(shape=(48,48,3))
    xavier = initializers.glorot_uniform()
    x = Conv2D(32,(3,3),kernel_initializer=xavier, name='conv1')(image_input)
    x = PReLU(name='prelu1')(x)
    x = MaxPooling2D((3,3),strides=(2,2), name='pool1')(x)

    x = Conv2D(64,(3,3),kernel_initializer=xavier, name='conv2')(x)
    x = PReLU(name='prelu2')(x)
    x = MaxPooling2D((3,3),strides=(2,2), name='pool2')(x)
   
    x = Conv2D(64,(3,3),kernel_initializer=xavier, name='conv3')(x)
    x = PReLU(name='prelu3')(x)
    x = MaxPooling2D((2,2),strides=(2,2), name='pool3')(x)

    x = Conv2D(128,(2,2),kernel_initializer=xavier, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(256,kernel_initializer=xavier,name='conv5')(x)
    x = Dropout(0.25)(x)
    x = PReLU(name='prelu5')(x)
    
    cla = Dense(1,activation=sigmoid,kernel_initializer=xavier,name='cla')(x)
    bbox = Dense(4,kernel_initializer=xavier,name='bbox')(x)
    landmark = Dense(10,kernel_initializer=xavier,name='landmark')(x)

    model = Model(inputs=image_input,outputs=[cla,bbox,landmark])

    return model

if __name__ == '__main__':
    PNet()