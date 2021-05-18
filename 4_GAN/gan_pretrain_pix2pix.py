# THIS CODE IS BASED ON https://github.com/phillipi/pix2pix

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.losses import binary_crossentropy
import numpy as np
import os
import tensorflow as tf
import albumentations as A
import math

# ---------------------------------- Define augmentations & global variables ---------------------------------

# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
augment = True

training_epochs = 2000
batch_size = 20
data_path = 'data/data.npz'

# Albumentations library composed augmentations

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5, rotate_limit=(-15, 15)),
    A.GridDistortion(p=0.2)],
    # A.RandomGamma(p=0.5)],
    additional_targets={'image1': 'image'})


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

# Function to generate mask / checkerboard
def checkerboard(w, h, c0, c1, blocksize):
    tile = np.array([[c0, c1], [c1, c0]]).repeat(
        blocksize, axis=0).repeat(blocksize, axis=1)
    grid = np.tile(tile, (int(math.ceil((h + 0.0) / (2 * blocksize))),
                          int(math.ceil((w + 0.0) / (2 * blocksize)))))
    return grid[:h, :w]

# define the discriminator model


def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    # opt = Adam(lr=0.0001, beta_1=0.5)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt, loss_weights=[0.5])
    # model.compile(loss=SSIMLoss, optimizer=opt, loss_weights=[0.5])

    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, block_name='name', batchnorm=True):
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


# define the standalone generator model
def define_generator(image_shape=(256, 256, 1)):
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
    g = Conv2DTranspose(1, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=[1, 100])
    return model


# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1 = data['arr_0']
    return [X1.astype('float32')]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape, mask):
    # unpack dataset
    trainA = dataset[0]
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    reference_image = trainA[ix]
    # # Augmentation
    if augment:
        for i in range(reference_image.shape[0]):
            augmented = aug(image=reference_image[i])
            reference_image[i] = augmented['image']
    # scale from [0,255] to [-1,1]
    reference_image = (reference_image - 127.5) / 127.5
    # Create a copy of reference image array and apply checkerboard mask
    masked_image = np.copy(reference_image)
    masked_image[mask == 1] = 1
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [masked_image, reference_image], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=4):
    # Create a checkerboard mask array
    mask = checkerboard(192, 192, 0, 1, 16)
    mask = np.repeat(mask[np.newaxis, :, :], n_samples, axis=0)
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1, mask)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(np.squeeze(X_realA[i], axis=2), cmap='gray')
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(np.squeeze(X_fakeB[i], axis=2), cmap='gray')
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(np.squeeze(X_realB[i], axis=2), cmap='gray')
    # save plot to file
    filename1 = 'weights/plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'weights/model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # Generate checkerboard mask image
    mask = checkerboard(192, 192, 0, 1, 16)
    mask = np.repeat(mask[np.newaxis, :, :], n_batch, axis=0)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(
            dataset, n_batch, n_patch, mask)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        if i % (100//n_batch) == 0:
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' %
                  (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo * 20) == 0:
            # if (i + 1) % 5 == 0:
            summarize_performance(i, g_model, dataset)

#                  __  __    _    ___ _   _
#                 |  \/  |  / \  |_ _| \ | |
#  _____   _____  | |\/| | / _ \  | ||  \| |  _____   _____
# |_____| |_____| | |  | |/ ___ \ | || |\  | |_____| |_____|
#                 |_|  |_/_/   \_\___|_| \_|
#


if __name__ == '__main__':
    # load image data
    dataset = load_real_samples('data/data.npz')
    print('Loaded', dataset[0].shape)
    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]
    print(image_shape)
    # define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    # define the composite model
    gan_model = define_gan(g_model, d_model, image_shape)
    # train model
    train(d_model, g_model, gan_model, dataset, n_epochs=2000, n_batch=20)
