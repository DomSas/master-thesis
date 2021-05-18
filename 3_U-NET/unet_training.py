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

sm.set_framework('tf.keras')


# GLOBAL VARIABLES
SEED = 1
smooth = 1.

img_rows = int(192)
img_cols = int(192)

BACKBONE = 'resnet50'

# TENSORBOARD DIRECTORY
tb_dir = 'logs/transfer_350epochs'


# INPUT PREPROCESSING FOR SEGMENTATION MODELS LIBRARY
preprocess_input = get_preprocessing(BACKBONE)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.float32)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


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
    imgs_train = np.load('../train/imgs.npy')
    masks_train = np.load('../train/masks.npy')
    return imgs_train, masks_train

def load_val_data():
    imgs_val = np.load('../val/imgs.npy')
    masks_val = np.load('../val/masks.npy')
    return imgs_val, masks_val


print('-'*20)
print('Loading the data')
print('-'*20)

imgs_train, masks_train = load_train_data()
imgs_val, masks_val = load_val_data()


##################################### PREPROCESS DATA #####################################
# all data are already normalized in range [0,1] and float32

# TRAIN
imgs_train = preprocess(imgs_train)
imgs_train = preprocess_input(imgs_train)

masks_train = preprocess(masks_train)
masks_train = preprocess_input(masks_train)

# VALIDATION
imgs_val = preprocess(imgs_val)
imgs_val = preprocess_input(imgs_val)

masks_val = preprocess(masks_val)
masks_val = preprocess_input(masks_val)


# SHUFFLE THE SLICES
shuffler_train = np.random.permutation(len(imgs_train))
shuffler_val = np.random.permutation(len(imgs_val))

imgs_train = imgs_train[shuffler_train]
masks_train = masks_train[shuffler_train]

imgs_val = imgs_val[shuffler_val]
masks_val = masks_val[shuffler_val]


##################################### AUGMENTATION OF TRAINING DATA #####################################


# DEFINE AUGMENTATION
albumentation_combo = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5, rotate_limit=(-15, 15)),
    A.GridDistortion(p=0.2)
])

img_data_gen = ImageDataAugmentor(
    augment=albumentation_combo,
    input_augment_mode='image',
    validation_split=0.2,
    seed=SEED,
)

mask_data_gen = ImageDataAugmentor(
    augment=albumentation_combo,
    input_augment_mode='mask',
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


##################################### AUGMENTATION OF VALIDATION DATA #####################################


val_imgs_gen = img_data_gen.flow(imgs_val, batch_size=32, subset='validation')
val_masks_gen = mask_data_gen.flow(
    masks_val, batch_size=32, subset='validation')

train_generator = zip(tr_img_gen, tr_mask_gen)
validation_generator = zip(val_imgs_gen, val_masks_gen)


##################################### DEFINE THE MODEL #####################################


N = imgs_train.shape[-1]  # number of channels

base_model = Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=False)

inp = Input(shape=(None, None, 1))
l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)


##################################### COMPILE THE MODEL #####################################

model.compile(optimizer=Adam(lr=4e-5), loss=Combo_loss, metrics=[dice_coef])


print('-'*20)
print('Training the model')
print('-'*20)

my_callbacks = [ModelCheckpoint('FIX_UNET_2.h5', verbose=1, monitor='val_dice_coef', save_best_only=True, mode='max'),
                tensorflow.keras.callbacks.TensorBoard(log_dir=tb_dir)]


history = model.fit(
    train_generator,
    steps_per_epoch=128,  # number of slices (2050) * 2 / batch size (32)
    epochs=300,
    validation_data=validation_generator,
    validation_steps=24,
    callbacks=[my_callbacks]
)


# SAVING GRAPH OF TRAINING+VALIDATION METRICS
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Model dice coeff')
plt.ylabel('Dice coeff')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('results_transfer_4e-5_100epochs.png')


print('-'*20)
print("Finished")
