import os
import numpy as np
import nibabel
from skimage.transform import resize
from skimage.io import imsave
from tensorflow.keras import backend as K

smooth = 1.

img_rows = int(192)
img_cols = int(192)

# LOADING THE DATA
predicted_path = './FIX_GAN_1.npy'

predicted = np.load(predicted_path)
original = np.load('./masks_test.npy')


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

def thresholding(image, input_range):
    image[image <= input_range] = 0
    image[image > input_range] = 1
    return image


orig = preprocess(orig)

# FIND BEST THRESHOLDING VALUE
best_dice = 0
initial_dice = float(dice_coef(orig, predicted))

print('Seaching best match for thresholding')
for threshold_value in range(0,100,1):    
    predicted = np.load(predicted_path)
    predicted_thresholded = thresholding(predicted, threshold_value/100)
    
    thesholded_dice = float(dice_coef(orig, predicted_thresholded))
    
    if(thesholded_dice > best_dice):
        best_dice_range = threshold_value
        best_dice = thesholded_dice

print('Best range is: ', best_dice_range/100)


# APPLY THRESHOLDING
predicted = np.load(predicted_path)
initial_dice = float(dice_coef(orig, predicted))

tresholded = thresholding(predicted, best_dice_range/100)
thresholded_dice = float(dice_coef(orig, tresholded))

print('Initial dice is', round(initial_dice, 3))
print('Thresholded dice is', round(thresholded_dice, 3))


# SAVING THE IMAGES
pred_dir = 'predicted_masks'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

for k in range(len(tresholded)):
    a=tresholded[k][:,:,0]
    imsave(os.path.join(pred_dir, str(k) + '_pred.png'), a) 