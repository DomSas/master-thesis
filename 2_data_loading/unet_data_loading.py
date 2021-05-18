import os
import cv2
import numpy as np
import nibabel


# NORMALIZE FUNCTION
def save_np_normalize(input_array):
    input_array = np.asarray(input_array)
    input_array /= input_array.max()
    input_array = input_array.astype('float32')

    return input_array


##################################### TRAIN DATA #####################################


# LOAD TRAINING PATHS
imgs_train_path = os.path.join('train/imgs')
masks_train_path = os.path.join('train/masks')

imgs_train = []
masks_train = []

nonzero_imgs_train = []
nonzero_masks_train = []


# TRAIN IMAGES
for dirr in sorted(os.listdir(imgs_train_path)):
    dirr = os.path.join(imgs_train_path, dirr)

    training_image = nibabel.load(dirr)
    for k in range(training_image.shape[2]):
        image_2d = np.array(training_image.get_fdata()[::, ::, k])
        imgs_train.append(image_2d)

print(len(imgs_train))


# TRAIN MASKS
for dirr in sorted(os.listdir(masks_train_path)):
    dirr = os.path.join(masks_train_path, dirr)

    training_mask = nibabel.load(dirr)
    for k in range(training_mask.shape[2]):
        mask_2d = np.array(training_mask.get_fdata()[::, ::, k])
        masks_train.append(mask_2d)

print(len(masks_train))


# FILTERING TRAINING SLICES
for i in range(len(imgs_train)):
    for j in range(192):
        if any(masks_train[i][j]):
            nonzero_imgs_train.append(imgs_train[i])
            nonzero_masks_train.append(masks_train[i])
            break


# NUMBER OF INITIAL VS FILTERED SLICES
print('Original size of train slices (imgs)')
print(len(imgs_train))
print('Size of train slices without zero slices (imgs)')
print(len(nonzero_imgs_train))

print('Original size of train slices (mask)')
print(len(masks_train))
print('Size of train slices without zero slices (mask)')
print(len(nonzero_masks_train))


# NORMALIZATION
nonzero_imgs_train = save_np_normalize(nonzero_imgs_train)
nonzero_masks_train = save_np_normalize(nonzero_masks_train)

print('-'*20)
print('Training images done')


##################################### VALIDATION DATA #####################################


# LOAD VALIDATION PATHS
imgs_val_path = os.path.join('val/imgs')
masks_val_path = os.path.join('val/masks')

imgs_val = []
masks_val = []

nonzero_imgs_val = []
nonzero_masks_val = []


# VALIDATION IMAGES
for dirr in sorted(os.listdir(imgs_val_path)):
    dirr = os.path.join(imgs_val_path, dirr)

    val_image = nibabel.load(dirr)
    for k in range(val_image.shape[2]):
        image_2d = np.array(val_image.get_fdata()[::, ::, k])
        imgs_val.append(image_2d)

print(len(imgs_val))


# VALIDATION MASKS
for dirr in sorted(os.listdir(masks_val_path)):
    dirr = os.path.join(masks_val_path, dirr)

    val_mask = nibabel.load(dirr)
    for k in range(val_mask.shape[2]):
        mask_2d = np.array(val_mask.get_fdata()[::, ::, k])
        masks_val.append(mask_2d)

print(len(masks_val))


# FILTERING VALIDATION SLICES
for i in range(len(imgs_val)):
    for j in range(192):
        if any(masks_val[i][j]):
            nonzero_imgs_val.append(imgs_train[i])
            nonzero_masks_val.append(masks_train[i])
            break


# NUMBER OF INITIAL VS FILTERED SLICES
print('Original size of validation slices (imgs)')
print(len(imgs_val))
print('Size of validation slices without zero slices (imgs)')
print(len(nonzero_imgs_val))

print('Original size of validation slices (mask)')
print(len(masks_val))
print('Size of validation slices without zero slices (mask)')
print(len(nonzero_masks_val))


# NORMALIZATION
nonzero_imgs_val = save_np_normalize(nonzero_imgs_val)
nonzero_masks_val = save_np_normalize(nonzero_masks_val)

print('-'*20)
print('Validation images done')


##################################### TEST DATA #####################################


# LOAD TEST PATHS
imgs_test_path = os.path.join('test/imgs')
masks_test_path = os.path.join('test/masks')

imgs_test = []
masks_test = []


# TEST IMAGES
for dirr in sorted(os.listdir(imgs_test_path)):
    dirr = os.path.join(imgs_test_path, dirr)

    testing_image = nibabel.load(dirr)
    for k in range(testing_image.shape[2]):
        image_2d = np.array(testing_image.get_fdata()[::, ::, k])
        imgs_test.append(image_2d)

print(len(imgs_test))


# TEST MASKS
for dirr in sorted(os.listdir(masks_test_path)):
    dirr = os.path.join(masks_test_path, dirr)

    testing_mask = nibabel.load(dirr)
    for k in range(testing_mask.shape[2]):
        mask_2d = np.array(testing_mask.get_fdata()[::, ::, k])
        masks_test.append(mask_2d)

print(len(masks_test))

# Test slices will not be filtered to simulate real data input

# NORMALIZATION
imgs_test = save_np_normalize(imgs_test)
masks_test = save_np_normalize(masks_test)

print('-'*20)
print('Test images done')


##################################### SAVING #####################################


print('-'*20)
print('Saving the data')

np.save('./nonzero_imgs_train.npy', nonzero_imgs_train)
np.save('./nonzero_masks_train.npy', nonzero_masks_train)

np.save('./nonzero_imgs_val.npy', nonzero_imgs_val)
np.save('./nonzero_masks_val.npy', nonzero_masks_val)

np.save('./imgs_test.npy', imgs_test)
np.save('./masks_test.npy', masks_test)

print('-'*20)
print('Saving is done')
