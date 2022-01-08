import numpy as np
import pydicom as dcm
import nibabel as nib
import os

def write_nii(voxels, output_path): # writes nii.gz file with header like and ITK-Snap segmentation
    niiheader = nib.nifti1.Nifti1Header()
    niiheader['bitpix'] = np.array(16, dtype=np.int16)
    niiheader['xyzt_units'] = np.array(2, dtype=np.uint8)
    niiheader['datatype'] = np.array(512, dtype=np.int16)# This MEANS 'uint16'
    niiheader['dim'] = np.array([3, voxels.shape[1], voxels.shape[2], voxels.shape[0], 1, 1, 1, 1], dtype=np.int16)
    nib.save(nib.Nifti1Image(np.transpose(voxels, [1, 2, 0]), np.eye(4), header=niiheader), output_path)

def voxels_from_nii(path_to_file):
    img = nib.load(path_to_file)
    voxels = img.get_fdata().astype(np.int16)
    return np.transpose(voxels, [2, 0, 1])

def voxels_from_dicom_folder(folder_path, d_type=np.int16):
    file_list = os.listdir(folder_path)
    first_dcm_path = os.path.join(folder_path, file_list[0])
    first_dcm = dcm.read_file(first_dcm_path)
    voxels = np.zeros([first_dcm.pixel_array.shape[0], first_dcm.pixel_array.shape[1], len(file_list)], dtype=d_type)
    for file_name in file_list:
        if file_name[-4:] == '.dcm':
            i = int(file_name[-8:-4])
            path = os.path.join(folder_path, file_name)
            dicom = dcm.read_file(path)
            voxels[:,:,i] = dicom.pixel_array

    return np.transpose(voxels, [2, 1, 0])
        
def index_grid(vol):
    y, x, z = np.meshgrid(np.arange(vol.shape[1]), np.arange(vol.shape[0]), np.arange(vol.shape[2]))
    return x, y, z

def mask_erode(mask, radius):
    y_kern, x_kern, z_kern = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    kern = (x_kern**2 + y_kern**2 + z_kern**2) < radius**2
    return sig.convolve(mask*1, kern, mode='same') == np.sum(kern)

def mask_dilate(mask, radius):
    y_kern, x_kern, z_kern = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    kern = (x_kern**2 + y_kern**2 + z_kern**2) < radius**2
    return sig.convolve(mask*1, kern, mode='same') > 0


if __name__ == '__main__':
    # Maybe this should run tests?
    print('You\'re really supposed to import this file, not run it.')
