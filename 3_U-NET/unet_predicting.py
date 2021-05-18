import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import argparse
import albumentations as A
import os

from skimage.io import imsave
from skimage.transform import resize

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import binary_crossentropy, DiceLoss
from segmentation_models import get_preprocessing

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from ImageDataAugmentor.image_data_augmentor import *
from skimage.exposure import rescale_intensity


# GLOBAL VARIABLES
BACKBONE = 'resnet50'
METRICS = DiceLoss
LOSS = 'binary_crossentropy'


# INPUT PREPROCESSING FOR SEGMENTATION MODELS LIBRARY
preprocess_input = get_preprocessing(BACKBONE)


def load_test_data():
    imgs_test = np.load('../data40_npy_loaded_WithVal/40_imgs_test.npy')
    return imgs_test


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


# LOAD TEST DATA
imgs_test = load_test_data()

imgs_test = preprocess(imgs_test)
imgs_test = preprocess_input(imgs_test)
imgs_test = rescale_intensity(imgs_test[:][:,:,:],out_range=(0,1))


##################################### DEFINE THE MODEL #####################################


# DEFINE THE MODEL
N = imgs_test.shape[-1]  # number of channels

base_model = Unet(BACKBONE, encoder_weights='imagenet')

inp = Input(shape=(None, None, 1))
l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)


##################################### PREDICT #####################################


# LOAD PRE-TRAINED WEIGHTS
model.load_weights('weights_transfer_4e-5_100epochs.h5')

# PREDICT
imgs_mask_test = model.predict(imgs_test, verbose=1)

# SAVE PREDICTED NPY ARRAY
np.save('predicted_masks.npy', imgs_mask_test)
