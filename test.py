from __future__ import print_function, division


from keras import backend as K
from keras.applications import VGG19
from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from scipy.misc import imsave, imread, imresize
import numpy as np
import os
import time
import warnings
import tensorflow as tf
from numpy import linalg as LA
import os
import scipy.io as sio 

def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def mat_upscale(mat_path):
    global model
    data_path = r"C:\Users\verohu\Desktop\feature-driven\perceptual_loss_test\test_data"
    name = mat_path
	
    patch = sio.loadmat(name)['x']
    img_conv = patch
    
    inputs = np.zeros((1, 32, 32, 3))
    inputs[0,:,:,:] = img_conv
#        model = self.create_model(load_weights=True)
    
    result =  model.predict(inputs)
    result = result[0]
    
	
    path = mat_path
    filename = path[:-4] + "_scaled.mat"
    
    sio.savemat(os.path.join(data_path, filename), {'result':result})
    
input_shape= (32, 32, 3)
print(input_shape)

def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
vgg = VGG19(weights="imagenet")
vgg.outputs = [vgg.layers[9].output]
img = Input(shape=(128, 128, 3))
img_features = vgg(img)
model_vgg = Model(img, img_features)
model_vgg.trainable = False
for l in model_vgg.layers:
    l.trainable = False

def _residual_block(ip, id):
        init = ip
        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)
        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

def _upscale_block(ip, id):
    init = ip
    x = UpSampling2D()(init)
    x = UpSampling2D()(x)
    x = Convolution2D(64, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)
    return x

def model_test(load_weights=True):
    input_shape = (32, 32, 3)
    init = Input(shape = input_shape)
    x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)
    x = _residual_block(x0, 1)
    x = Add()([x, x0])
    x = _upscale_block(x, 1)
    x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)
    model = Model(init, x)
    adam = optimizers.Adam(lr=5e-4)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
    weight_path = "test.hdf5" 
#model = model.load_weights(weight_path, by_name=True)
    if load_weights: model.load_weights(weight_path, by_name=True)
    model = model
    return model
#x_vgg = model_vgg(model.outputs)
#full_model = Model(model.inputs, x_vgg);   #针对x

model = model_test(load_weights = True)
with tf.device('/GPU:0'):
    path = r"C:\Users\verohu\Desktop\feature-driven\perceptual_loss_test\test_data";
    files = os.listdir(path)
    n = 0;
    for p in files:
        f = path+"\\"+p
        mat_upscale(f)
        n=n+1
        print(n)


    
