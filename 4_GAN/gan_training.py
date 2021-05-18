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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.losses import binary_crossentropy
from matplotlib import pyplot
from skimage.transform import resize
from ImageDataAugmentor.image_data_augmentor import *
from skimage.exposure import rescale_intensity

import numpy as np
import os
import tensorflow
import albumentations as A
import math
import matplotlib.pyplot as plt

# TENSORBOARD DIRECTORY
tb_dir = 'logs/GAN_with_predictions_freezed'

img_rows = int(192)
img_cols = int(192)

smooth = 1.
SEED=1

e = K.epsilon()
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss


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


def Combo_loss(targets, inputs):
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
                
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, e, 1.0 - e)
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
                                            
    return combo


def load_train_data():
    imgs_train = np.load('../data40_npy_loaded_WithVal_nonzero/40_nonzero_imgs_train.npy')
    masks_train = np.load('../data40_npy_loaded_WithVal_nonzero/40_nonzero_masks_train.npy')
    return imgs_train, masks_train

def load_val_data():
    imgs_val = np.load('../data40_npy_loaded_WithVal_nonzero/40_nonzero_imgs_val.npy')
    masks_val = np.load('../data40_npy_loaded_WithVal_nonzero/40_nonzero_masks_val.npy')
    return imgs_val, masks_val

def load_test_data():
    imgs_test = np.load('../data40_npy_loaded_WithVal/40_imgs_test.npy')
    return imgs_test


##################################### FUNCTIONS TO CREATE THE MODEL #####################################


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


##################################### LOAD THE DATA #####################################


# LOADING THE DATA
imgs_train, masks_train = load_train_data()
imgs_val, masks_val = load_val_data()

# PREPROCESS THE DATA
imgs_train = preprocess(imgs_train)
masks_train = preprocess(masks_train)

imgs_val = preprocess(imgs_val)
masks_val = preprocess(masks_val)


# SHUFFLE THE SLICES
shuffler_train = np.random.permutation(len(imgs_train))
shuffler_val = np.random.permutation(len(imgs_val))

imgs_train = imgs_train[shuffler_train]
masks_train = masks_train[shuffler_train]

imgs_val = imgs_val[shuffler_val]
masks_val = masks_val[shuffler_val]

# LOAD TEST DATA
imgs_test = load_test_data()
imgs_test = preprocess(imgs_test)

##################################### AUGMENTATION OF TRAINING DATA #####################################



import albumentations as A
from ImageDataAugmentor.image_data_augmentor import *
SEED = 1
N_CLASSES = 2
rgb_weights = [0.2989, 0.5870, 0.1140]

albumentation_combo = A.Compose([
   	A.VerticalFlip(p=0.5),
	A.RandomRotate90(p=0.5),
	A.ShiftScaleRotate(p=0.5, rotate_limit=(-15, 15)),
	A.GridDistortion(p=0.2)
#     A.ShiftScaleRotate(p=1)
    ])


##################################### Training #####################################


img_data_gen = ImageDataAugmentor(
    augment=albumentation_combo,
    input_augment_mode='image',
    validation_split=0.2,
    seed=SEED,
)

mask_data_gen = ImageDataAugmentor(
    augment=albumentation_combo,
    input_augment_mode='mask', #<- notice the different augment mode
#     preprocess_input=one_hot_encode_masks,
    validation_split=0.2,
    seed=SEED,
)

# RESCALE INTENSITY OF IMAGES
imgs_train = rescale_intensity(imgs_train[:][:,:,:],out_range=(0,1))
masks_train = rescale_intensity(masks_train[:][:,:,:],out_range=(0,1))

imgs_val = rescale_intensity(imgs_val[:][:,:,:],out_range=(0,1))
masks_val = rescale_intensity(masks_val[:][:,:,:],out_range=(0,1))


tr_img_gen = img_data_gen.flow(imgs_train, batch_size=32, subset='training')
tr_mask_gen = mask_data_gen.flow(masks_train, batch_size=32, subset='training')


##################################### Validation #####################################


val_imgs_gen = img_data_gen.flow(imgs_val, batch_size=32, subset='validation')
val_masks_gen = mask_data_gen.flow(masks_val, batch_size=32, subset='validation')

train_generator = zip(tr_img_gen, tr_mask_gen)
validation_generator = zip(val_imgs_gen, val_masks_gen)


##################################### CREATE THE MODEL #####################################


model = define_transfer_generator_freezed()
model.compile(optimizer=Adam(lr=2e-5),
              loss=Combo_loss, metrics=[dice_coef])

my_callbacks = [ModelCheckpoint('FIX_GAN_3.h5', verbose=1, monitor='val_dice_coef', save_best_only=True, mode='max')]

# LOAD PRE-TRAINED WEIGHTS
model.load_weights('GAN_orig_weights.h5', by_name=True, skip_mismatch=True)


history = model.fit(
    train_generator,
    steps_per_epoch=128,  # number of slices (2050) * 2 / batch size (32)
    epochs=300,
    validation_data=validation_generator,
    validation_steps=24,
    callbacks=[my_callbacks]
)


# LOAD TRAINED WEIGTS
model.load_weights('FIX_GAN_3.h5', by_name=True)

# PREDICT FROM THE MODEL
imgs_mask_test = model.predict(imgs_test, verbose=1)
np.save('FIX_GAN_3.npy', imgs_mask_test)
print("Predicting done")


# SAVING GRAPH OF TRAINING+VALIDATION METRICS
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Model dice coeff')
plt.ylabel('Dice coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
plt.savefig('FIX_GAN_3.png')
# plotting our dice coeff results in function of the number of epochs


print('-'*20)
print("Finished")
