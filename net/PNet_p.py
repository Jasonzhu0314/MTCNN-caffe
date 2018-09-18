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
    xavier = initializers.glorot_normal()

    x = Conv2D(10,(3,3), kernel_initializer=xavier, name='conv1')(image_input)
    x = PReLU(name='prelu1',shared_axes=[1,2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(16,(3,3), kernel_initializer=xavier, name='conv2')(x)
    x = PReLU(name='prelu2',shared_axes=[1,2])(x)

    x = Conv2D(32,(3,3), kernel_initializer=xavier, name='conv3')(x)
    x = PReLU(name='prelu3',shared_axes=[1,2])(x)

    cla = Conv2D(1,(1,1), kernel_initializer=xavier, name='cla',activation=sigmoid)(x)
    
    bbox = Conv2D(4,(1,1), kernel_initializer=xavier, name='bbox')(x)

    model = Model(inputs=image_input,outputs=[cla,bbox])
    
    return model

def PNet_cla():
    image_input = Input(shape=(None,None,3))
    xavier = initializers.glorot_normal()

    x = Conv2D(10,(3,3), kernel_initializer=xavier, name='conv1')(image_input)
    x = PReLU(name='prelu1',shared_axes=[1,2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(16,(3,3), kernel_initializer=xavier, name='conv2')(x)
    x = PReLU(name='prelu2',shared_axes=[1,2])(x)

    x = Conv2D(32,(3,3), kernel_initializer=xavier, name='conv3')(x)
    x = PReLU(name='prelu3',shared_axes=[1,2])(x)

    cla = Conv2D(1,(1,1), kernel_initializer=xavier, name='cla',activation=sigmoid)(x)

    model = Model(inputs=image_input,outputs=cla)
    
    return model
def PNet_cla1():
    image_input = Input(shape=(None,None,3))
    xavier = initializers.glorot_normal()

    x = Conv2D(10,(3,3), kernel_initializer=xavier, name='conv1')(image_input)
    x = PReLU(name='prelu1',shared_axes=[1,2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(16,(3,3), kernel_initializer=xavier, name='conv2')(x)
    x = PReLU(name='prelu2',shared_axes=[1,2])(x)

    x = Conv2D(32,(3,3), kernel_initializer=xavier, name='conv3')(x)
    x = PReLU(name='prelu3',shared_axes=[1,2])(x)

    cla = Conv2D(1,(1,1), kernel_initializer=xavier, name='cla',activation=sigmoid)(x)

    bbox = Conv2D(4,(1,1), kernel_initializer=xavier, name='bbox')(x)

    model = Model(inputs=image_input,outputs=[cla,bbox])
    
    return model
if __name__ == '__main__':
    model = PNet()
    model.summary()