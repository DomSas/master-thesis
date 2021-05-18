import os
import SimpleITK as sitk


# CHANGE .raw FORMAT FILES TO .nii
def miccai_to_nifti(input_folder, output_folder):
    """ Function to loop through folders of MICCAI dataset and convert all to nifti files """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for dirr in sorted(os.listdir(input_folder)):
        input_directory = os.path.join(input_folder, dirr)
        output_directory = os.path.join(output_folder, dirr)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        nifti_files = sorted(os.listdir(input_directory))
        for scan in nifti_files:
            if scan.endswith(".nhdr"):
                input_nifti_path = os.path.join(input_directory, scan)
                if 'T1' in scan:
                    output_nifti_path = os.path.join(
                        output_directory, 'T1.nii')
                if 'T2' in scan:
                    output_nifti_path = os.path.join(
                        output_directory, 'T2.nii')
                if 'FLAIR' in scan:
                    output_nifti_path = os.path.join(
                        output_directory, 'flair.nii')
                if 'lesion' in scan:
                    output_nifti_path = os.path.join(
                        output_directory, 'mask.nii')
                img_raw = sitk.ReadImage(input_nifti_path)
                sitk.WriteImage(img_raw, output_nifti_path)


##################################### MAIN #####################################


if __name__ == '__main__':
    input_nifti_folder = './sample/miccai/'
    output_nifti_folder = './sample/miccai_nifti/'
    miccai_to_nifti(input_nifti_folder, output_nifti_folder)
