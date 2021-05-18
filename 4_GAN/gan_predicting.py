from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
from tensorflow.keras.losses import binary_crossentropy
from skimage.transform import resize
from skimage.exposure import rescale_intensity

import numpy as np
import os
import tensorflow as tf
import albumentations as A
import math
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

smooth = 1.
img_rows = int(192)
img_cols = int(192)


def load_test_data():
    imgs_test = np.load('./masks_test.npy')
    return imgs_test


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#########################################################################


# define an encoder block
def define_encoder_block(layer_in, n_filters, block_name='name', batchnorm=True, trainable=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same',
            kernel_initializer=init, name='conv_'+block_name)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization(name='batch_'+block_name)(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, block_name='name', dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init, name='conv_'+block_name)(layer_in)
    # add batch normalization
    g = BatchNormalization(name='batch_'+block_name)(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# GENERATOR WITH ORIGINAL ENCODER PART
def define_original_generator(image_shape=(192, 192, 1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, block_name='e1', batchnorm=False)
    e2 = define_encoder_block(e1, 128, block_name='e2')
    e3 = define_encoder_block(e2, 256, block_name='e3')
    e4 = define_encoder_block(e3, 512, block_name='e4')
    e5 = define_encoder_block(e4, 512, block_name='e5')
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init, name='bottleneck')(e5)
    b = Activation('relu')(b)
    # decoder model
    d3 = decoder_block(b, e5, 512, block_name='d1')
    d4 = decoder_block(d3, e4, 512, block_name='d2', dropout=False)
    d5 = decoder_block(d4, e3, 256, block_name='d3', dropout=False)
    d6 = decoder_block(d5, e2, 128, block_name='d4', dropout=False)
    d7 = decoder_block(d6, e1, 64, block_name='d5', dropout=False)
    # output
    g = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    g2 = Conv2D(1, (1, 1), padding='same')(g)
    out_image = Activation('sigmoid')(g2)
    # define model
    model = Model(in_image, out_image)
    return model


# GENERATOR WITH RENAMED DECODER PART
def define_transfer_generator(image_shape=(192, 192, 1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, block_name='e1', batchnorm=False)
    e2 = define_encoder_block(e1, 128, block_name='e2')
    e3 = define_encoder_block(e2, 256, block_name='e3')
    e4 = define_encoder_block(e3, 512, block_name='e4')
    e5 = define_encoder_block(e4, 512, block_name='e5')
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init, name='bottleneck_RENAMED')(e5)
    b = Activation('relu')(b)
    # decoder model
    d3 = decoder_block(b, e5, 512, block_name='d1_RENAMED')
    d4 = decoder_block(d3, e4, 512, block_name='d2_RENAMED', dropout=False)
    d5 = decoder_block(d4, e3, 256, block_name='d3_RENAMED', dropout=False)
    d6 = decoder_block(d5, e2, 128, block_name='d4_RENAMED', dropout=False)
    d7 = decoder_block(d6, e1, 64, block_name='d5_RENAMED', dropout=False)
    # output
    g = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    g2 = Conv2D(1, (1, 1), padding='same')(g)
    out_image = Activation('sigmoid')(g2)
    # define model
    model = Model(in_image, out_image)
    return model


# GENERATOR WITH FREEZED ENCODER PART
def define_transfer_generator_freezed(image_shape=(192, 192, 1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, block_name='e1', batchnorm=False, trainable=False)
    e2 = define_encoder_block(e1, 128, block_name='e2', trainable=False)
    e3 = define_encoder_block(e2, 256, block_name='e3', trainable=False)
    e4 = define_encoder_block(e3, 512, block_name='e4', trainable=False)
    e5 = define_encoder_block(e4, 512, block_name='e5', trainable=False)

    # FREEZE THE LAYERS
    e1.trainable = False
    e2.trainable = False
    e3.trainable = False
    e4.trainable = False
    e5.trainable = False

    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init, name='bottleneck_RENAMED')(e5)
    b = Activation('relu')(b)
    # decoder model
    d3 = decoder_block(b, e5, 512, block_name='d1_RENAMED')
    d4 = decoder_block(d3, e4, 512, block_name='d2_RENAMED', dropout=False)
    d5 = decoder_block(d4, e3, 256, block_name='d3_RENAMED', dropout=False)
    d6 = decoder_block(d5, e2, 128, block_name='d4_RENAMED', dropout=False)
    d7 = decoder_block(d6, e1, 64, block_name='d5_RENAMED', dropout=False)
    # output
    g = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    g2 = Conv2D(1, (1, 1), padding='same')(g)
    out_image = Activation('sigmoid')(g2)
    # define model
    model = Model(in_image, out_image)
    return model


#########################################################################

# LOAD TEST DATA
imgs_test = load_test_data()
imgs_test = preprocess(imgs_test)
imgs_test = rescale_intensity(imgs_test[:][:,:,:],out_range=(0,1))


# CREATING THE MODEL
model = define_transfer_generator()

# LOAD PRE-TRAINED WEIGTS
model.load_weights('weights.h5')

# PREDICT FROM THE MODEL
imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('predicted_masks.npy', imgs_mask_test)
