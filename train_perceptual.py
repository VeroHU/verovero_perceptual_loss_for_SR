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

K.set_image_dim_ordering('tf') # tf: (height, width, channels) ,th:  (channels, height, width)
X_train  = np.load("train_X.npy")
X_validate  = np.load("vali_X.npy")

y_train  = np.load("train_y.npy")
y_validate  = np.load("vali_y.npy")



input_shape= X_train[0].shape
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

input_shape = (32, 32, 3)
init = Input(shape = input_shape)
x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)
x = _residual_block(x0, 1)
x = Add()([x, x0])
x = _upscale_block(x, 1)
x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)
model = Model(init, x)


x_vgg = model_vgg(model.outputs)
full_model = Model(model.inputs, x_vgg);   #针对x
y_vgg = model_vgg.predict(y_train)  #针对y
y_vgg_vali = model_vgg.predict(y_validate)  #针对y


adam = optimizers.Adam(lr=5e-4)
full_model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss])
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, 
                              min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, 
                              save_best_only=True)
history = full_model.fit(X_train, y_vgg, 
                    batch_size=64, 
                    epochs=100, 
                    verbose=1, 
                    validation_data=(X_validate, y_vgg_vali),
                    callbacks=[reduce_lr, checkpointer],
                    shuffle=True)