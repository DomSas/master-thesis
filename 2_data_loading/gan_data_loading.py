import nibabel as nib
import os
import numpy as np
from numpy import savez_compressed

data_path = 'data/train'
output_file = 'data/data.npz'
img_size = 192

input_array = np.zeros((0, img_size, img_size, 1))


# NORMALIZE FUNCTION
def normalize(img_numpy):
    MAX, MIN = img_numpy.max(), img_numpy.min()
    img_numpy = (img_numpy - MIN) * 255 / (MAX - MIN)
    return img_numpy


for dirr in sorted(os.listdir(data_path)):
    print(dirr)
    input_dirr = os.path.join(data_path, dirr)
    for nifti in sorted(os.listdir(input_dirr)):
        input_img = nib.load(input_dirr + '/' + nifti)

        input_data = np.transpose(input_img.get_fdata(), (2, 0, 1))
        input_data = normalize(input_data)
        input_array = np.concatenate(
            (input_array, np.expand_dims(input_data, axis=3)), 0)
        print(input_array.shape)


# SAVING DATASET
savez_compressed(output_file, input_array)
print('-'*20)
print('Saved dataset: ', output_file)
