import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np


# FUNCTION TO DISPLAY ROW OF IMAGE SLICES
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


# FUNCTION TO VISUALISE 3 SLICES FROM NIFTI IMAGE AND PRINT BASIC INFO
def print_scan(nifti):
    print("Input data shape is: " + str(nifti.shape))
    print("Input max value is: " + str(nifti.max()))
    print("Input min value is: " + str(nifti.min()))
    print("Input mean value is: " + str(nifti.mean()))
    slice_0 = nifti[nifti.shape[0] // 2, :, :]
    slice_1 = nifti[:, nifti.shape[1] // 2, :]
    slice_2 = nifti[:, :, nifti.shape[2] // 2]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices of the input nifti image")
    plt.show()


# FUNCTION TO PROCESS SCANS AND MASKS FROM ISBI DATASET
def process_isbi_scan_mask(input_path, output_path):
    # Load input scan
    img = nib.load(input_path)
    data = img.get_fdata()
    # Apply next two lines to resample to 192x192x192
    data = data[:, 12:204, :]
    data = np.pad(data, ((5, 6), (0, 0), (5, 6)), 'constant')
    # Print output volume
    # print_scan(data)
    # Normalize between 0 - 255
    # data += np.abs(data.min())
    # data *= 255.0 / data.max()
    output_nifti = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(output_nifti, output_path)


# FUNCTION TO PROCESS SCANS FROM BOTH MICCAI AND MSSEG DATASETS
def process_miccai_msseg_scan(input_path, output_path):
    img = nib.load(input_path)
    # resampled_nii = nib_proc.resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0))
    resampled_nii = nib_proc.conform(img, out_shape=(
        192, 192, 192), voxel_size=(1.0, 1.0, 1.0), orientation="LPS")
    data = resampled_nii.get_fdata()
    data -= data.min()
    # print_scan(data)
    output_nifti = nib.Nifti1Image(
        data, resampled_nii.affine, resampled_nii.header)
    nib.save(output_nifti, output_path)


# FUNCTION TO PROCESS MASKS FROM BOTH MICCAI AND MSSEG DATASETS
def process_miccai_msseg_mask(input_path, output_path):
    img = nib.load(input_path)
    resampled_nii = nib_proc.conform(img, out_shape=(
        192, 192, 192), voxel_size=(1, 1, 1), orientation="LPS")
    data = resampled_nii.get_fdata()
    # Normalize between 0 - 1
    data -= data.min()
    data *= 1.0 / data.max()
    # Threshold masks
    # np.where(data > 0.5, 1, 0)
    data[data > 0.5] = 1.0
    data[data < 0.5] = 0.0

    # print_scan(data)
    output_nifti = nib.Nifti1Image(
        data, resampled_nii.affine, resampled_nii.header)
    nib.save(output_nifti, output_path)


# FUNCTION TO LOOP THROUGH FOLDERS AND APPLY RESAMPLING FUNCTION ON NIFTI FILES
def loop_folders(input_folder, output_folder, function):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for dirr in sorted(os.listdir(input_folder)):
        input_directory = os.path.join(input_folder, dirr)
        output_directory = os.path.join(output_folder, dirr)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        nifti_files = sorted(os.listdir(input_directory))
        for scan in nifti_files:
            input_nifti_path = os.path.join(input_directory, scan)
            output_nifti_path = os.path.join(output_directory, scan)
            print('Processing scan ' + str(input_nifti_path))
            function(input_nifti_path, output_nifti_path)


##################################### MAIN #####################################


if __name__ == '__main__':
    # Process isbi scans
    # input_nifti_folder = './sample/train/isbi'
    # output_nifti_folder = './sample/train_resampled/isbi'
    # loop_folders(input_nifti_folder, output_nifti_folder, process_isbi_scan_mask)

    # Process isbi masks
    input_nifti_folder = './sample/masks/isbi'
    output_nifti_folder = './sample/masks_resampled/isbi'
    loop_folders(input_nifti_folder, output_nifti_folder,
                 process_isbi_scan_mask)

    # # Process miccai scans
    # input_nifti_folder = './sample/train/miccai'
    # output_nifti_folder = './sample/train_resampled/miccai'
    # loop_folders(input_nifti_folder, output_nifti_folder, process_miccai_msseg_scan)

    # # Process miccai masks
    # input_nifti_folder = './sample/masks/miccai'
    # output_nifti_folder = './sample/masks_resampled/miccai'
    # loop_folders(input_nifti_folder, output_nifti_folder, process_miccai_msseg_mask)

    # # Process msseg scans
    # input_nifti_folder = './sample/train/msseg'
    # output_nifti_folder = './sample/train_resampled/msseg'
    # loop_folders(input_nifti_folder, output_nifti_folder, process_miccai_msseg_scan)

    # # Process msseg masks
    # input_nifti_folder = './sample/masks/msseg'
    # output_nifti_folder = './sample/masks_resampled/msseg'
    # loop_folders(input_nifti_folder, output_nifti_folder, process_miccai_msseg_mask)
