import numpy as np
import pydicom as dcm
import os
import scipy
import cc3d
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.signal as sig
from numpy import linalg

import xradct

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Sitka'], 'variant':'small-caps'})

np.seterr(divide='ignore') # Ignore divide by zero warnings.

# Region_growing
def onion_grow(x, y, z, initial_mask, container_mask, kernel_radius, n_iters, return_label_sizes=False, largest_only=False, verbose=False):
    # This is roughly the same as the region growing portion of head_segmentation.rainbow_skeleton()
    assert x.shape==y.shape==z.shape==initial_mask.shape==container_mask.shape

    if np.sum(initial_mask)==0:
        print('warning!!: onion_grow recieved empty initial_mask')

    growth_kernel = spherical_kernel(kernel_radius)

    # Region growing is calculated only over the range defined by these bounds:
    xmin = np.maximum(0, np.min(x[initial_mask])-growth_kernel.shape[0])
    xmax = np.minimum(initial_mask.shape[0]-1, np.max(x[initial_mask])+growth_kernel.shape[0])
    ymin = np.maximum(0, np.min(y[initial_mask])-growth_kernel.shape[1])
    ymax = np.minimum(initial_mask.shape[1]-1, np.max(y[initial_mask])+growth_kernel.shape[1])
    zmin = np.maximum(0, np.min(z[initial_mask])-growth_kernel.shape[2])
    zmax = np.minimum(initial_mask.shape[2]-1, np.max(z[initial_mask])+growth_kernel.shape[2])

    seg = -np.ones(initial_mask.shape, dtype=np.int32)
    seg[container_mask] = 0
    seg[initial_mask] = 1
    fuzzy_mask = np.clip(initial_mask.astype(np.float32), 0, 1)

    if return_label_sizes:
        label_sizes = [np.sum(initial_mask)]

    for i in range(2, n_iters+1):
        fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax] = np.clip(
            scipy.signal.fftconvolve(fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax], growth_kernel, mode='same'),
            0, 1)*(seg[xmin:xmax,ymin:ymax,zmin:zmax] == 0)
        if largest_only:
            new_label_mask = largest_connected_only((seg[xmin:xmax,ymin:ymax,zmin:zmax] == 0)&(fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax] > .5))
            fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax][~new_label_mask] = 0
        else:
            fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax][fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax]<.01] = 0 # mitigates compounding numerical errors
            new_label_mask = (seg[xmin:xmax,ymin:ymax,zmin:zmax] == 0)&(fuzzy_mask[xmin:xmax,ymin:ymax,zmin:zmax] > .5)

        if ~np.any(new_label_mask):
            if verbose:
                print(f'Region growing halted at iteration {i}.')
            break
        else:
            if return_label_sizes:
                label_sizes.append(np.sum(new_label_mask))
            seg[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask] = i

            xmin, xmax, ymin, ymax, zmin, zmax = (
                np.maximum(0, np.min(x[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask])-growth_kernel.shape[0]),
                np.minimum(initial_mask.shape[0]-1, np.max(x[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask])+growth_kernel.shape[0]),
                np.maximum(0, np.min(y[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask])-growth_kernel.shape[1]),
                np.minimum(initial_mask.shape[1]-1, np.max(y[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask])+growth_kernel.shape[1]),
                np.maximum(0, np.min(z[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask])-growth_kernel.shape[2]),
                np.minimum(initial_mask.shape[2]-1, np.max(z[xmin:xmax,ymin:ymax,zmin:zmax][new_label_mask])+growth_kernel.shape[2])
                )
    if return_label_sizes:
        return seg, np.array(label_sizes)
    else:
        return seg

def largest_connected_only(mask, connectivity=26):
    cc3d_labels = cc3d.connected_components(mask, connectivity=connectivity)
    cc3d_stats = cc3d.statistics(cc3d_labels)
    voxel_counts = cc3d_stats['voxel_counts']
    sorted_indices = np.argsort(voxel_counts)[::-1]
    for index in sorted_indices:
        candidate_mask = (cc3d_labels==index)
        if np.any(mask[candidate_mask]):
            return candidate_mask

    print('largest_connected_only() Failed!!')
    return mask

def second_largest_connected_only(mask, connectivity=26):
    cc3d_labels = cc3d.connected_components(mask, connectivity=connectivity)
    cc3d_stats = cc3d.statistics(cc3d_labels)
    voxel_counts = cc3d_stats['voxel_counts']
    sorted_indices = np.argsort(voxel_counts)[::-1]

    first_skipped = False
    for index in sorted_indices:
        candidate_mask = (cc3d_labels==index)
        if np.any(mask[candidate_mask]):
            if not first_skipped:
                first_skipped = True
            else:
                return candidate_mask

    print('second_largest_connected_only() Failed!!')
    return mask

def exclude_small_connected(mask, threshold_volume, connectivity=6):
    output_mask = np.zeros(mask.shape, dtype=bool)
    cc3d_labels = cc3d.connected_components(mask, connectivity=connectivity)
    cc3d_stats = cc3d.statistics(cc3d_labels)
    voxel_counts = cc3d_stats['voxel_counts']
    sorted_indices = np.argsort(voxel_counts)[::-1]
    for index in sorted_indices:
        candidate_mask = (cc3d_labels==index)
        if np.any(mask[candidate_mask])&(voxel_counts[index]>threshold_volume):
            output_mask |= candidate_mask

    return output_mask

# Image Filtering
def gaussian_filter_mm(sigma_mm, voxels_per_mm, extent_mm):
    x_values = np.arange(-extent_mm/voxels_per_mm[0], extent_mm/voxels_per_mm[0]+1)
    y_values = np.arange(-extent_mm/voxels_per_mm[1], extent_mm/voxels_per_mm[1]+1)
    z_values = np.arange(-extent_mm/voxels_per_mm[2], extent_mm/voxels_per_mm[2]+1)

    y_grid, x_grid, z_grid = np.meshgrid(y_values, x_values, z_values)
    sigma_voxels = sigma_mm/voxels_per_mm

    kernel = np.exp(-x_grid**2/sigma_voxels[0]**2 - y_grid**2/sigma_voxels[1]**2 - z_grid**2/sigma_voxels[2]**2)
    return kernel

def reg_anchor_filt(voxels, sigma_env_mm, extent_sigma=2, detail_scale_mm=1, voxels_per_mm=np.array([.2,.2,.2]), pad_val=0):
    extent_mm = sigma_env_mm*extent_sigma

    x_values = np.arange(-int(np.round(extent_mm/voxels_per_mm[0])), int(np.round(extent_mm/voxels_per_mm[0]+1)))
    y_values = np.arange(-int(np.round(extent_mm/voxels_per_mm[1])), int(np.round(extent_mm/voxels_per_mm[1]+1)))
    z_values = np.arange(-int(np.round(extent_mm/voxels_per_mm[2])), int(np.round(extent_mm/voxels_per_mm[2]+1)))

    y_grid, x_grid, z_grid = np.meshgrid(y_values, x_values, z_values)
    sigma_env_voxels = sigma_env_mm/voxels_per_mm

    env_filt = np.exp(-x_grid**2/sigma_env_voxels[0]**2-y_grid**2/sigma_env_voxels[1]**2-z_grid**2/sigma_env_voxels[2]**2)
    env_filt /= np.sum(env_filt)*1000
    detail_filt_length = (np.round(detail_scale_mm/voxels_per_mm)).astype(int)*2+1
    detail_filt_param_x = np.linspace(-np.pi*(1-1/detail_filt_length[0]), np.pi*(1-1/detail_filt_length[0]), num=detail_filt_length[0], endpoint=True)
    detail_filt_param_y = np.linspace(-np.pi*(1-1/detail_filt_length[1]), np.pi*(1-1/detail_filt_length[1]), num=detail_filt_length[1], endpoint=True)
    detail_filt_param_z = np.linspace(-np.pi*(1-1/detail_filt_length[2]), np.pi*(1-1/detail_filt_length[2]), num=detail_filt_length[2], endpoint=True)

    detail_filt_x = (np.sin(detail_filt_param_x)[:,np.newaxis,np.newaxis]*
                     np.cos(detail_filt_param_y/2)[np.newaxis,:,np.newaxis]*
                     np.cos(detail_filt_param_z/2)[np.newaxis,np.newaxis,:])
    #detail_filt_x /= np.sum(np.abs(detail_filt_x))
    detail_filt_y = (np.cos(detail_filt_param_x/2)[:,np.newaxis,np.newaxis]*
                     np.sin(detail_filt_param_y)[np.newaxis,:,np.newaxis]*
                     np.cos(detail_filt_param_z/2)[np.newaxis,np.newaxis,:])
    #detail_filt_y /= np.sum(np.abs(detail_filt_y))
    detail_filt_z = (np.cos(detail_filt_param_x/2)[:,np.newaxis,np.newaxis]*
                     np.cos(detail_filt_param_y/2)[np.newaxis,:,np.newaxis]*
                     np.sin(detail_filt_param_z)[np.newaxis,np.newaxis,:])
    #detail_filt_z /= np.sum(np.abs(detail_filt_z))

    x_stick = scipy.signal.convolve(np.abs(scipy.signal.convolve(voxels, detail_filt_x, mode='same')), env_filt, mode='same')/10
    y_stick = scipy.signal.convolve(np.abs(scipy.signal.convolve(voxels, detail_filt_y, mode='same')), env_filt, mode='same')/10
    z_stick = scipy.signal.convolve(np.abs(scipy.signal.convolve(voxels, detail_filt_z, mode='same')), env_filt, mode='same')/10

    output = x_stick*y_stick*z_stick
    output /= np.max(output)

    return output

def spherical_window(blank):
    x, y, z = index_grid(blank)
    x_center = (blank.shape[0]-1)/2
    y_center = (blank.shape[1]-1)/2
    z_center = (blank.shape[2]-1)/2

    r_squared = distance_squared_from_point(x, y, z, [x_center, y_center, z_center])
    max_r_squared = np.max([x_center, y_center, z_center])**2

    return np.maximum(0, 1-r_squared/max_r_squared)

def spherical_kernel(radius):
    # This implementation can only take an interger radius... It could use some work.
    y, x, z = np.meshgrid(np.arange(-radius, radius+1)/(radius+1/radius), np.arange(-radius, radius+1)/(radius+1/radius), np.arange(-radius, radius+1)/(radius+1/radius))
    r_squared = (x*x+y*y+z*z)
    return np.clip(1-r_squared, 0, 1)

def inverse_square_kernel(radius, center_val=1):
    # note that "radius" is kind of a poorly named argument. It is
    y, x, z = np.meshgrid(np.arange(-radius, radius+1)/(radius+1/radius), np.arange(-radius, radius+1)/(radius+1/radius), np.arange(-radius, radius+1)/(radius+1/radius))
    inv_r_squared = 1/(x*x+y*y+z*z).astype(np.float32)
    inv_r_squared[np.isinf(inv_r_squared)] = center_val
    return inv_r_squared

# Region Registration
def correlation_offset(base_layer, slip_layer, search_dist, search_center, patch_size):
    base_x_min = np.maximum(0, search_center[0]-search_dist-patch_size//2)
    base_x_max = np.minimum(base_x_min + patch_size + search_dist*2, base_layer.shape[0]-1)
    base_y_min = np.maximum(0, search_center[1]-search_dist-patch_size//2)
    base_y_max = np.minimum(base_y_min + patch_size + search_dist*2, base_layer.shape[1]-1)
    base_z_min = np.maximum(0, search_center[2]-search_dist-patch_size//2)
    base_z_max = np.minimum(base_z_min + patch_size + search_dist*2, base_layer.shape[2]-1)

    slip_x_min = np.maximum(0, search_center[0]-patch_size//2)
    slip_x_max = np.minimum(slip_x_min + patch_size, slip_layer.shape[0]-1)
    slip_y_min = np.maximum(0, search_center[1]-patch_size//2)
    slip_y_max = np.minimum(slip_y_min + patch_size, slip_layer.shape[1]-1)
    slip_z_min = np.maximum(0, search_center[2]-patch_size//2)
    slip_z_max = np.minimum(slip_z_min + patch_size, slip_layer.shape[2]-1)

    base_block = base_layer[base_x_min:base_x_max, base_y_min:base_y_max, base_z_min:base_z_max]
    slip_block = slip_layer[slip_x_min:slip_x_max, slip_y_min:slip_y_max, slip_z_min:slip_z_max]

    if np.sum(slip_block) < 1000:
        print('empty slip block')
        return 0, 0, 0, np.nan

    slip_block = spherical_window(slip_block)*(slip_block)
    #print(base_block.shape, slip_block.shape)
    xcorr = scipy.signal.correlate(base_block, slip_block, mode='valid')
    xcorr /= scipy.signal.correlate(base_block, np.ones(slip_block.shape), mode='valid')
    #print(xcorr.shape)
    raw_i, raw_j, raw_k = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    #voxelhelp.view_xyz_mips(xcorr, annotation=[(raw_i, raw_j, raw_k)])
    best_base_block = base_block[raw_i:raw_i+slip_block.shape[0], raw_j:raw_j+slip_block.shape[1], raw_k:raw_k+slip_block.shape[2]]

    view_center_slices(slip_block, CT_scale=False)
    view_center_slices(best_base_block, CT_scale=False)
    return raw_i-search_dist, raw_j-search_dist, raw_k-search_dist, np.max(xcorr)/np.sum(xcorr)

def offset_of_min_error(base_layer, slip_layer, search_dist, search_center, patch_size, view_blocks=False):
    base_x_min = np.maximum(0, search_center[0]-search_dist-patch_size//2)
    base_x_max = np.minimum(base_x_min + patch_size + search_dist*2, base_layer.shape[0]-1)
    base_y_min = np.maximum(0, search_center[1]-search_dist-patch_size//2)
    base_y_max = np.minimum(base_y_min + patch_size + search_dist*2, base_layer.shape[1]-1)
    base_z_min = np.maximum(0, search_center[2]-search_dist-patch_size//2)
    base_z_max = np.minimum(base_z_min + patch_size + search_dist*2, base_layer.shape[2]-1)

    slip_x_min = np.maximum(0, search_center[0]-patch_size//2)
    slip_x_max = np.minimum(slip_x_min + patch_size, slip_layer.shape[0]-1)
    slip_y_min = np.maximum(0, search_center[1]-patch_size//2)
    slip_y_max = np.minimum(slip_y_min + patch_size, slip_layer.shape[1]-1)
    slip_z_min = np.maximum(0, search_center[2]-patch_size//2)
    slip_z_max = np.minimum(slip_z_min + patch_size, slip_layer.shape[2]-1)

    slip_block = slip_layer[slip_x_min:slip_x_max, slip_y_min:slip_y_max, slip_z_min:slip_z_max]
    window = spherical_window(slip_block)
    window_mean = np.mean(window)

    if np.sum(slip_block) < 1000:
        print('empty slip block')
        return 0, 0, 0, np.nan

    x_offset_min = base_x_min - slip_x_min
    x_offset_max = base_x_max - slip_x_max
    y_offset_min = base_y_min - slip_y_min
    y_offset_max = base_y_max - slip_y_max
    z_offset_min = base_z_min - slip_z_min
    z_offset_max = base_z_max - slip_z_max

    i_best, j_best, k_best = 0, 0, 0
    best_error = np.inf
    error_sum = 0

    for i in range(x_offset_min, x_offset_max):
        for j in range(y_offset_min, y_offset_max):
            for k in range(z_offset_min, z_offset_max):
                difference = slip_block - base_layer[slip_x_min+i:slip_x_max+i, slip_y_min+j:slip_y_max+j, slip_z_min+k:slip_z_max+k]
                error = np.sqrt(np.mean(difference*difference*window)/window_mean)
                error_sum += error
                if error < best_error:
                    best_error = error
                    i_best, j_best, k_best = i, j, k

    if view_blocks:
        best_base_block = base_layer[slip_x_min+i_best:slip_x_max+i_best, slip_y_min+j_best:slip_y_max+j_best, slip_z_min+k_best:slip_z_max+k_best]
        view_center_slices(slip_block, CT_scale=False)
        view_center_slices(best_base_block, CT_scale=False)

    return i_best, j_best, k_best, -(best_error/(np.mean(slip_block*window)**2)*1000)

def volume_peaks(volume):
    # This fucntion is not finished...
    left = np.roll(volume,1,axis=0)
    right = np.roll(volume,-1,axis=0)
    forward = np.roll(volume,1,axis=1)
    backward = np.roll(volume,-1,axis=1)
    up = np.roll(volume,1,axis=2)
    down = np.roll(volume,-1,axis=2)

    xyz_max = np.maximum(np.maximum(np.maximum(left, right), np.maximum(forward, backward)), np.maximum(up, down))
    is_local_max = (volume>=xyz_max)
    print(np.sum(is_local_max))

def adjust_point_pair_by_registration(voxels, x, y, z, point_a, point_b, patch_size, search_dist, view_blocks=False):
    # similar to offset_of_min_error(), but does not require previously computed mirrored volume
    base_x_min = np.maximum(0, point_a[0]-search_dist-patch_size//2)
    base_x_max = np.minimum(base_x_min + patch_size + search_dist*2, voxels.shape[0]-1)
    base_y_min = np.maximum(0, point_a[1]-search_dist-patch_size//2)
    base_y_max = np.minimum(base_y_min + patch_size + search_dist*2, voxels.shape[1]-1)
    base_z_min = np.maximum(0, point_a[2]-search_dist-patch_size//2)
    base_z_max = np.minimum(base_z_min + patch_size + search_dist*2, voxels.shape[2]-1)

    slip_x_min = point_a[0]-patch_size//2
    slip_x_max = slip_x_min + patch_size
    slip_y_min = point_a[1]-patch_size//2
    slip_y_max = slip_y_min + patch_size
    slip_z_min = point_a[2]-patch_size//2
    slip_z_max = slip_z_min + patch_size

    source_slip_y, source_slip_x, source_slip_z = np.meshgrid(
        np.arange(slip_y_min, slip_y_max),
        np.arange(slip_x_min, slip_x_max),
        np.arange(slip_z_min, slip_z_max)
        )

    point_A, point_B = np.array(point_a), np.array(point_b)
    center = (point_A + point_B)/2
    normal = (point_B - point_A) # vector normal to the plane equidistant to the points
    normal = normal/np.sqrt(np.sum(normal*normal)) # Nomalizing the normal vector

    slip_x, slip_y, slip_z = mirror_coordinates(source_slip_x, source_slip_y, source_slip_z, center, normal)
    slip_block = resamp_volume(voxels, slip_x, slip_y, slip_z, mode='linear', default_val=0)
    window = spherical_window(slip_block)*~(slip_block==0)
    window_mean = np.mean(window)

    if np.sum(slip_block) < 1000:
        print('empty slip block')
        return point_a, point_b

    x_offset_min = base_x_min - slip_x_min
    x_offset_max = base_x_max - slip_x_max
    y_offset_min = base_y_min - slip_y_min
    y_offset_max = base_y_max - slip_y_max
    z_offset_min = base_z_min - slip_z_min
    z_offset_max = base_z_max - slip_z_max

    i_best, j_best, k_best = 0, 0, 0
    best_error = np.inf
    #error_sum = 0

    for i in range(x_offset_min, x_offset_max):
        for j in range(y_offset_min, y_offset_max):
            for k in range(z_offset_min, z_offset_max):
                difference = slip_block - voxels[slip_x_min+i:slip_x_max+i, slip_y_min+j:slip_y_max+j, slip_z_min+k:slip_z_max+k]
                error = np.sqrt(np.mean(difference*difference*window)/window_mean)
                #error_sum += error
                if error < best_error:
                    best_error = error
                    i_best, j_best, k_best = i, j, k

    if view_blocks:
        best_base_block = voxels[slip_x_min+i_best:slip_x_max+i_best, slip_y_min+j_best:slip_y_max+j_best, slip_z_min+k_best:slip_z_max+k_best]
        print('Slip Block')
        view_center_slices(slip_block, CT_scale=False)
        print(f'Corresponding source voxels offset by <{i_best}, {j_best}, {k_best}>')
        view_center_slices(best_base_block, CT_scale=False)

    return (point_a[0]+i_best, point_a[1]+j_best, point_a[2]+k_best), point_b

# Reading and writing medical image files
def write_nii(voxels, output_path, voxel_spacing=[.2,.2,.2]): # writes nii.gz file with header like and ITK-Snap segmentation
    if not os.path.exists(os.path.split(output_path)[0]):
        #print(f'output_path {os.path.split(output_path)[0]} created.')
        os.makedirs(os.path.split(output_path)[0])
    niiheader = nib.nifti1.Nifti1Header()
    niiheader['bitpix'] = np.array(16, dtype=np.int16)
    niiheader['xyzt_units'] = np.array(2, dtype=np.uint8)
    niiheader['datatype'] = np.array(512, dtype=np.int16)# This MEANS 'uint16'
    niiheader['dim'] = np.array([3, voxels.shape[1], voxels.shape[2], voxels.shape[0], 1, 1, 1, 1], dtype=np.int16)
    niiheader['pixdim'] = np.array([1, voxel_spacing[1], voxel_spacing[2], voxel_spacing[0], 0, 0, 0, 0], dtype=np.float32)
    nib.save(nib.Nifti1Image(np.transpose(voxels.astype(np.int16), [1, 2, 0]), np.eye(4), header=niiheader), output_path)

def voxels_from_nii(path_to_file):
    img = nib.load(path_to_file)
    voxels = img.get_fdata().astype(np.int16)
    return np.transpose(voxels, [2, 0, 1])

def voxels_from_dicom_folder(folder_path, d_type=np.int16):
    file_list = os.listdir(folder_path)
    for file in file_list:
        if file[-4:] == '.dcm':
            first_dcm_path = os.path.join(folder_path, file)
            break
    else:
        raise RuntimeError(f'voxels_from_dicom_folder() did not find .dcm files in {folder_path}.')

    first_dcm = dcm.dcmread(first_dcm_path, force=True)
    xz_spacing = np.array(first_dcm.PixelSpacing)
    y_spacing = np.array(first_dcm.SliceThickness)
    voxel_spacing = [xz_spacing[0], y_spacing, xz_spacing[1]]
    voxels = np.zeros([first_dcm.pixel_array.shape[0], first_dcm.pixel_array.shape[1], len(file_list)], dtype=d_type)
    date_taken = first_dcm.StudyDate
    for file_name in file_list:
        if file_name[-4:] == '.dcm':
            i = int(file_name[-8:-4])-1
            path = os.path.join(folder_path, file_name)
            dicom = dcm.read_file(path)
            voxels[:,:,i] = dicom.pixel_array

    return np.transpose(voxels, [1, 2, 0]), voxel_spacing, date_taken

def nudge_CT_numbers(voxels, hist_smooth_width=12, show_plots=False, no_standards=True):
    counts, bins = np.histogram(voxels, bins=500)
    bin_means = (bins[:-1]+bins[1:])/2
    smoothing_kernel = scipy.signal.convolve(np.ones(hist_smooth_width),np.ones(hist_smooth_width))
    smoothing_kernel /= np.sum(smoothing_kernel)

    smoothed_counts = scipy.signal.convolve(counts, smoothing_kernel, mode='same')
    higher_peak = bin_means[np.argmax(smoothed_counts)]
    not_too_close = np.abs(bin_means-higher_peak)>700
    lower_peak = bin_means[np.argmax(smoothed_counts*not_too_close)]

    air_value = np.minimum(higher_peak, lower_peak)
    tissue_value = np.maximum(higher_peak, lower_peak)

    if ((air_value<-700)&(air_value>-1300)&(tissue_value<300)&(tissue_value>-300))|no_standards:
        corrected_voxels = (1030/(tissue_value-air_value))*(voxels.astype(np.float32)-tissue_value)+30
        if show_plots:
            plt.plot(bin_means, smoothed_counts)
            plt.plot((1000/(tissue_value-air_value))*(bin_means-tissue_value), smoothed_counts)
            plt.show()
        return corrected_voxels
    else:
        print('nudge_CT_numbers() recieved wird histogram and was not able to fix it.')
        print(f'air value: {air_value}, tissue value: {tissue_value}')
        if show_plots:
            plt.plot(bin_means, smoothed_counts)
            plt.title('Voxel Histogram')
            plt.show()
        return voxels


# 3D Geometry utilities
def index_grid(vol):
    # The fact that the argument to this function is the volume instead of the shape was a mistake that should probably be corrected...
    y, x, z = np.meshgrid(np.arange(vol.shape[1]), np.arange(vol.shape[0]), np.arange(vol.shape[2]))
    return x, y, z

def double_resolution(vol, target_shape=None):
    if target_shape == None:
        target_shape = np.array(vol.shape)*2
    big_vol = np.zeros(target_shape, dtype=vol.dtype)
    i, j, k = index_grid(big_vol)
    big_vol[i, j, k] = vol[np.clip(i//2, 0, vol.shape[0]-1), np.clip(j//2, 0, vol.shape[1]-1), np.clip(k//2, 0, vol.shape[2]-1)]
    return big_vol

def mask_index_centroid(mask): # finds indices of the center of a 3D mask image
    x, y, z = index_grid(mask)
    return np.mean(x[mask]).astype(int), np.mean(y[mask]).astype(int), np.mean(z[mask]).astype(int)

def mask_coordinate_centroid(mask, x, y, z):
    # similar to mask_index_centroid, but for arbitrary coordinate system
    # reduces peak memory compared to mask_index_centroid if and index grid is already stored
    assert x.shape==y.shape==z.shape
    return np.mean(x[mask]), np.mean(y[mask]), np.mean(z[mask])

def coordinate_sphere_mask(x, y, z, r, point):
    assert x.shape==y.shape==z.shape
    return (x-point[0])*(x-point[0]) + (y-point[1])*(y-point[1]) + (z-point[2])*(z-point[2]) < r*r

def closer_to_a_than_b(x, y, z, a, b):
    assert x.shape==y.shape==z.shape
    r_squared_a = (x-a[0])*(x-a[0]) + (y-a[1])*(y-a[1]) + (z-a[2])*(z-a[2])
    r_squared_b = (x-b[0])*(x-b[0]) + (y-b[1])*(y-b[1]) + (z-b[2])*(z-b[2])
    return r_squared_a < r_squared_b

def dist_a_to_b(a, b):
    return np.sqrt(np.sum([(a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2]))

def pins_and_string(x, y, z, foci, pseudoradius):
    # Given coordinate arrays x, y, & z, this function produces a binary image that is True where the sum of euclidian distances to each point in "foci" is less than "pseudoradius."
    assert x.shape==y.shape==z.shape
    rs = np.sqrt(np.array([(x-f[0])*(x-f[0]) + (y-f[1])*(y-f[1]) + (z-f[2])*(z-f[2]) for f in foci]))
    return np.sum(rs, axis=0) < pseudoradius

def tuple_mean(tuples):
    return tuple(np.mean([tup[pos] for tup in tuples]) for pos in range(len(tuples[0])))

def distance_squared_from_point(x, y, z, point):
    assert x.shape==y.shape==z.shape
    return np.maximum((x-point[0])*(x-point[0]) + (y-point[1])*(y-point[1]) + (z-point[2])*(z-point[2]), 1/100000000)

def distance_from_point(x, y, z, point):
    return np.sqrt(distance_squared_from_point(x, y, z, point))

def plane_near_points(points):
    center = np.mean(points, axis=0)
    normal = np.linalg.svd(points.T-np.outer(center, np.ones(points.shape[0])))[0][:,-1]
    return tuple(center), tuple(normal)

def three_points_to_normal(point_a, point_b, point_c):
    v_ab = np.array(point_b) - np.array(point_a)
    v_ac = np.array(point_c) - np.array(point_a)
    normal = np.cross(v_ab, v_ac)
    normal /= np.sqrt(np.sum(normal*normal))
    return (normal[0], normal[1], normal[2])

def pill_r_squared(x, y, z, point_a, point_b, min_val=1/1000000000):
    assert x.shape==y.shape==z.shape
    point_a = np.array(point_a).flatten()
    point_b = np.array(point_b).flatten()
    v_ab = np.array(point_b) - np.array(point_a)
    v_ab = v_ab/np.sqrt(np.sum(v_ab*v_ab))

    parallel_mag_a = (x-point_a[0])*v_ab[0] + (y-point_a[1])*v_ab[1] + (z-point_a[2])*v_ab[2]
    a_side = parallel_mag_a < 0
    b_side = (x-point_b[0])*v_ab[0] + (y-point_b[1])*v_ab[1] + (z-point_b[2])*v_ab[2] > 0

    ortho_x, ortho_y, ortho_z = -point_a[0] + x - parallel_mag_a*v_ab[0], -point_a[1] + y - parallel_mag_a*v_ab[1], -point_a[2] + z - parallel_mag_a*v_ab[2]
    ortho_r_squared = ortho_x*ortho_x + ortho_y*ortho_y + ortho_z*ortho_z
    a_r_squared = distance_squared_from_point(x, y, z, point_a)
    b_r_squared = distance_squared_from_point(x, y, z, point_b)

    return np.maximum(ortho_r_squared*~(a_side|b_side) + a_r_squared*a_side + b_r_squared*b_side, min_val)

# Coorinate mapping
def normal_field_from_point_pairs(x, y, z, point_pairs):
    assert x.shape==y.shape==z.shape

    normal_x_out = np.zeros(x.shape, dtype=np.float32)
    normal_y_out = np.zeros(y.shape, dtype=np.float32)
    normal_z_out = np.zeros(z.shape, dtype=np.float32)

    total_pull = np.zeros(x.shape, dtype=np.float64)
    for point_pair in point_pairs:
        point_a, point_b = np.array(point_pair[0]).flatten(), np.array(point_pair[1]).flatten()
        this_normal = (point_b - point_a)
        this_normal = this_normal/np.sqrt(np.sum(this_normal*this_normal))

        this_pull = 1/pill_r_squared(x, y, z, point_a, point_b)
        total_pull += this_pull

        current_weight = this_pull/total_pull

        normal_x_out = (1 - current_weight)*normal_x_out + current_weight*this_normal[0]
        normal_y_out = (1 - current_weight)*normal_y_out + current_weight*this_normal[1]
        normal_z_out = (1 - current_weight)*normal_z_out + current_weight*this_normal[2]

    mag_out = np.sqrt(normal_x_out*normal_x_out + normal_y_out*normal_y_out + normal_z_out*normal_z_out)

    return normal_x_out/mag_out, normal_y_out/mag_out, normal_z_out/mag_out

def mirror_coordinates(x, y, z, center, normal):
    assert x.shape==y.shape==z.shape
    centered_x = x - center[0]
    centered_y = y - center[1]
    centered_z = z - center[2]
    normal_mag = centered_x*normal[0] + centered_y*normal[1] + centered_z*normal[2]

    new_x = centered_x - 2*normal[0]*normal_mag + center[0]
    new_y = centered_y - 2*normal[1]*normal_mag + center[1]
    new_z = centered_z - 2*normal[2]*normal_mag + center[2]

    return new_x, new_y, new_z

def translate_with_point_pairs(x, y, z, point_pairs):
    assert x.shape==y.shape==z.shape
    x_out = np.zeros(x.shape, dtype=np.float32)
    y_out = np.zeros(y.shape, dtype=np.float32)
    z_out = np.zeros(z.shape, dtype=np.float32)

    centers = []
    normals = []
    for point_pair in point_pairs:
        point_A, point_B = np.array(point_pair[0]), np.array(point_pair[1])
        center = (point_A + point_B)/2
        normal = (point_B - point_A) # vector normal to the plane equidistant to the points
        normal = normal/np.sqrt(np.sum(normal*normal)) # Nomalizing the normal vector
        #print(center)
        #print(normal)
        centers.append(center)
        normals.append(normal)

    # The following would flip any normals that are pointing in the opposite direct; may not be necessary.
    if False:
        normals_array = np.array(normals)
        average_normal = np.mean(normals_array, axis=0)
        alignedness = np.dot(normals_array, average_normal)
        normals_array *= np.sign(alignedness)[:,np.newaxis]
        normals = [normals_array[n,:] for n in range(len(normals))]

    total_pull = np.zeros(x.shape, dtype=np.float64)
    A_pull = np.zeros(x.shape, dtype=np.float64)
    B_pull = np.zeros(x.shape, dtype=np.float64)
    for point_pair, center, normal in zip(point_pairs, centers, normals):

        point_A, point_B = np.array(point_pair[0]), np.array(point_pair[1])
        this_pull_A = 1/(.1+distance_squared_from_point(x, y, z, point_A).astype(np.float64))
        this_pull_B = 1/(.1+distance_squared_from_point(x, y, z, point_B).astype(np.float64))
        #this_pull_C = np.minimum(1/distance_squared_from_point(x, y, z, center).astype(np.float64), 99999999999999)*10
        this_pull = this_pull_A + this_pull_B# + this_pull_C
        total_pull += this_pull
        A_pull += this_pull_A
        B_pull += this_pull_B
        current_weight = this_pull/total_pull

        new_x, new_y, new_z = mirror_coordinates(x, y, z, center, normal)

        x_out = (1-current_weight)*x_out + current_weight*new_x
        y_out = (1-current_weight)*y_out + current_weight*new_y
        z_out = (1-current_weight)*z_out + current_weight*new_z
        #print(f'current weight: {current_weight}')


    return x_out, y_out, z_out, A_pull>B_pull

# Binary image processing
def mask_erode(mask, radius):
    # Old implementation
    #y_kern, x_kern, z_kern = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    #kern = x_kern**2 + y_kern**2 + z_kern**2 - radius**2
    #return sig.convolve(mask*1, kern, mode='same') == np.sum(kern)

    kern = spherical_kernel(radius)
    return sig.convolve(1-mask, kern, mode='same') < .5*np.sum(kern)

def mask_dilate(mask, radius):
    # Old implementation that gave blocky results
    #y_kern, x_kern, z_kern = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    #kern = (x_kern**2 + y_kern**2 + z_kern**2) < radius**2
    #return sig.convolve(mask*1, kern, mode='same') > 0

    return sig.convolve(mask*1, spherical_kernel(radius), mode='same') > .5

def mask_smooth(mask, radius, method='erode-dilate'):
    if method == 'erode-dilate':
        y_kern, x_kern, z_kern = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), np.arange(-radius, radius+1))
        kern = x_kern**2 + y_kern**2 + z_kern**2 - radius**2
        eroded = sig.convolve(mask*1, kern, mode='same') == np.sum(kern)
        return sig.convolve(eroded*1, kern, mode='same') > 0.001

    elif method == 'smoothie':
        kernel_fade = .75
        y_kern, x_kern, z_kern = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), np.arange(-radius, radius+1))
        kern = np.clip(radius - np.sqrt(x_kern**2 + y_kern**2 + z_kern**2), -.75, .75) + .75 #avoids kernel aliasing
        eroded = (sig.convolve(mask*1, kern, mode='same') > .5*np.sum(kern))
        return (sig.convolve(mask*1, kern, mode='same') > .5)

# Volume Resampling
def resamp_volume(voxels, x, y, z, mode='linear', default_val=-1000):
    # This function makes a new array of voxels by sampling an old array, "voxels," at coordinates x, y, and z, which can be arrays of floats.
    assert x.shape==y.shape==z.shape, f'resamp_volume recieved x, y, z with non-matching shapes {x.shape}, {y.shape}, and {z.shape}.'

    output = np.zeros(x.shape)

    temp_corner_val = voxels[0,0,0]
    voxels[0,0,0] = default_val

    if mode=='linear':
        x_base = np.floor(x).astype(int)
        y_base = np.floor(y).astype(int)
        z_base = np.floor(z).astype(int)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x_index = x_base+i
                    y_index = y_base+j
                    z_index = z_base+k
                    outside = (x_index<0)|(y_index<0)|(z_index<0)|(x_index>=voxels.shape[0]-1)|(y_index>=voxels.shape[1]-1)|(z_index>=voxels.shape[2]-1)
                    x_index[outside] = 0
                    y_index[outside] = 0
                    z_index[outside] = 0
                    elements = voxels[x_index, y_index, z_index]
                    weights = (1-np.abs(np.floor(x)+i-x))*(1-np.abs(np.floor(y)+j-y))*(1-np.abs(np.floor(z)+k-z))
                    output += elements*weights

    if mode=='nearest':
        x_index = np.round(x).astype(int)
        y_index = np.round(y).astype(int)
        z_index = np.round(z).astype(int)
        outside = (x_index<0)|(y_index<0)|(z_index<0)|(x_index>=voxels.shape[0]-1)|(y_index>=voxels.shape[1]-1)|(z_index>=voxels.shape[2]-1)
        x_index[outside] = 0
        y_index[outside] = 0
        z_index[outside] = 0
        output = voxels[x_index, y_index, z_index]

    voxels[0,0,0] = temp_corner_val

    return output

def resamp_with_inverse(voxels, x, y, z, mode='linear', default_val=-1000):
    # Not finished; do not use.
    assert x.shape==y.shape==z.shape, f'resamp_volume recieved x, y, z with non-matching shapes {x.shape}, {y.shape}, and {z.shape}.'

    x_inv = np.zeros(voxels.shape, dtype=np.float32)
    y_inv = np.zeros(voxels.shape, dtype=np.float32)
    z_inv = np.zeros(voxels.shape, dtype=np.float32)
    
    output = np.zeros(x.shape)

    x_out, y_out, z_out = index_grid(output)

    temp_corner_val = voxels[0,0,0]
    voxels[0,0,0] = default_val

    if mode=='linear':
        total_weight = np.zeros(voxels.shape, dtype=np.float32)

        x_base = np.floor(x).astype(int)
        y_base = np.floor(y).astype(int)
        z_base = np.floor(z).astype(int)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x_index = x_base+i
                    y_index = y_base+j
                    z_index = z_base+k
                    outside = (x_index<0)|(y_index<0)|(z_index<0)|(x_index>=voxels.shape[0]-1)|(y_index>=voxels.shape[1]-1)|(z_index>=voxels.shape[2]-1)
                    x_index[outside] = 0
                    y_index[outside] = 0
                    z_index[outside] = 0
                    elements = voxels[x_index, y_index, z_index]
                    weights = (1-np.abs(np.floor(x)+i-x))*(1-np.abs(np.floor(y)+j-y))*(1-np.abs(np.floor(z)+k-z))
                    output += elements*weights

                    x_inv[x_index, y_index, z_index] += weights*x_out
                    y_inv[x_index, y_index, z_index] += weights*y_out
                    z_inv[x_index, y_index, z_index] += weights*z_out
                    total_weight[x_index, y_index, z_index] += weights
        x_inv[x_inv>0] /= total_weight[x_inv>0]
        x_inv[x_inv<=0] = -1
        y_inv[y_inv>0] /= total_weight[y_inv>0]
        y_inv[y_inv<=0] = -1
        z_inv[z_inv>0] /= total_weight[z_inv>0]
        z_inv[z_inv<=0] = -1

    elif mode=='nearest':
        x_index = np.round(x).astype(int)
        y_index = np.round(y).astype(int)
        z_index = np.round(z).astype(int)
        outside = (x_index<0)|(y_index<0)|(z_index<0)|(x_index>=voxels.shape[0]-1)|(y_index>=voxels.shape[1]-1)|(z_index>=voxels.shape[2]-1)
        x_index[outside] = 0
        y_index[outside] = 0
        z_index[outside] = 0
        output = voxels[x_index, y_index, z_index]
        x_inv[x_index, y_index, z_index] = x_out
        y_inv[x_index, y_index, z_index] = y_out
        z_inv[x_index, y_index, z_index] = z_out

    voxels[0,0,0] = temp_corner_val

    return output, x_inv, y_inv, z_inv

def resamp_with_gradient(voxels, x, y, z, default_val=-1000):
    # This function does the same thing as resamp_volume(..., mode='linear'), but also computes derivatives of the output w.r.t. sampling locations.

    assert x.shape==y.shape==z.shape, f'resamp_volume recieved x, y, z with non-matching shapes {x.shape}, {y.shape}, and {z.shape}.'
    output = np.zeros(x.shape, dtype=np.float32)
    partial_x = np.zeros(x.shape, dtype=np.float32)
    partial_y = np.zeros(x.shape, dtype=np.float32)
    partial_z = np.zeros(x.shape, dtype=np.float32)

    temp_corner_val = voxels[0,0,0]
    voxels[0,0,0] = default_val

    x_base = np.floor(x).astype(np.int32)
    y_base = np.floor(y).astype(np.int32)
    z_base = np.floor(z).astype(np.int32)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x_index = x_base+i
                y_index = y_base+j
                z_index = z_base+k
                outside = (x_index<0)|(y_index<0)|(z_index<0)|(x_index>=voxels.shape[0]-1)|(y_index>=voxels.shape[1]-1)|(z_index>=voxels.shape[2]-1)
                x_index[outside] = 0
                y_index[outside] = 0
                z_index[outside] = 0
                elements = voxels[x_index, y_index, z_index]
                weights = (1-np.abs(np.floor(x)+i-x))*(1-np.abs(np.floor(y)+j-y))*(1-np.abs(np.floor(z)+k-z))
                output += elements*weights

                partial_x += elements*(2*i-1)
                partial_y += elements*(2*j-1)
                partial_z += elements*(2*k-1)

    voxels[0,0,0] = temp_corner_val

    return output, partial_x, partial_y, partial_z

def test_checker(x, y, z, cube_size=10):
    assert x.shape==y.shape==z.shape

    return np.mod(x//cube_size + y//cube_size + z//cube_size, 2)

def coordinate_translate(x_mirror, y_mirror, z_mirror, x_point, y_point, z_point):
    # Finds the argument indices into the coorinate system defined by x_mirror, y_mirror, z_mirror that would give <x_point, y_point, z_point>.
    # x_point, y_point, z_point can also be arrays, not necessarily a single point; Can be used to find the inverse coordinate transformation.

    # but how to actually implement it...
    pass

# Image Display
def view_xyz_mips(voxels, cmap='gray', ann_points=[], ann_labels=[], annotation=[], label_points=False, save_path=None, colorbars=False):
    if type(annotation)==xradct.PointAnnotationSet:
        ann_points = annotation.points
        ann_labels = annotation.labels
    elif type(annotation)==list:
        ann_points = annotation

    default_text_color = plt.rcParams['text.color']
    plt.rcParams.update({'text.color': "white"})
    if not save_path==None:
        fig = plt.figure(figsize=[11, 3], dpi=600)
    else:
        fig = plt.figure(figsize=[18,7])
    (x_disp, y_disp, z_disp) =  fig.subplots(1,3)#, gridspec_kw={'width_ratios': [voxels.shape[1]+50, voxels.shape[0]+50, voxels.shape[0]+50]})

    imx = x_disp.imshow((np.max(voxels, axis=0).T), cmap=cmap)
    x_disp.invert_yaxis()
    x_disp.set_xlabel('y')
    x_disp.set_ylabel('z')
    imy = y_disp.imshow((np.max(voxels, axis=1).T), cmap=cmap)
    y_disp.invert_yaxis()
    y_disp.set_xlabel('x')
    y_disp.set_ylabel('z')
    imz = z_disp.imshow((np.max(voxels, axis=2).T), cmap=cmap)
    z_disp.invert_yaxis()
    z_disp.set_xlabel('x')
    z_disp.set_ylabel('y')
    if colorbars:
        plt.colorbar(imx, ax=x_disp)
        plt.colorbar(imy, ax=y_disp)
        plt.colorbar(imz, ax=z_disp)
    for point in ann_points:
        x_disp.scatter(point[1], point[2])
        y_disp.scatter(point[0], point[2])
        z_disp.scatter(point[0], point[1])

    if label_points:
        for point, label in zip(ann_points, ann_labels):

            x_disp.annotate(label, (point[1], point[2]), fontsize=5)
            y_disp.annotate(label, (point[0], point[2]), fontsize=5)
            z_disp.annotate(label, (point[0], point[1]), fontsize=5)
    fig.tight_layout()
    if not save_path==None:
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.rcParams.update({'text.color': default_text_color})

def view_xyz_means(voxels, ann_points=[], ann_labels=[], annotation=[], label_points=False, save_path=None, colorbars=False):
    if type(annotation)==xradct.PointAnnotationSet:
        ann_points = annotation.points
        ann_labels = annotation.labels
    elif type(annotation)==list:
        ann_points = annotation

    
    default_text_color = plt.rcParams['text.color']
    plt.rcParams.update({'text.color': "white"})
    if not save_path==None:
        fig = plt.figure(figsize=[11, 3], dpi=600)
    else:
        fig = plt.figure(figsize=[18,7])
    (x_disp, y_disp, z_disp) =  fig.subplots(1,3, gridspec_kw={'width_ratios': [voxels.shape[1], voxels.shape[0], voxels.shape[0]]})

    imx = x_disp.imshow((np.mean(voxels, axis=0).T), cmap='gray')
    x_disp.invert_yaxis()
    x_disp.set_xlabel('y')
    x_disp.set_ylabel('z')
    imy = y_disp.imshow((np.mean(voxels, axis=1).T), cmap='gray')
    y_disp.invert_yaxis()
    y_disp.set_xlabel('x')
    y_disp.set_ylabel('z')
    imz = z_disp.imshow((np.mean(voxels, axis=2).T), cmap='gray')
    z_disp.invert_yaxis()
    z_disp.set_xlabel('x')
    z_disp.set_ylabel('y')
    if colorbars:
        plt.colorbar(imx, ax=x_disp)
        plt.colorbar(imy, ax=y_disp)
        plt.colorbar(imz, ax=z_disp)
    for point in ann_points:
        x_disp.scatter(point[1], point[2])
        y_disp.scatter(point[0], point[2])
        z_disp.scatter(point[0], point[1])
    if label_points:
        for point, label in zip(ann_points, ann_labels):
            x_disp.annotate(label, (point[1], point[2]), color='#FEFEFE')
            y_disp.annotate(label, (point[0], point[2]), color='#FEFEFE')
            z_disp.annotate(label, (point[0], point[1]), color='#FEFEFE')

    if not save_path==None:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.rcParams.update({'text.color': default_text_color})

def view_center_slices(voxels, CT_scale=True, cmap='gray', ann_points=[], ann_labels=[], annotation=[], label_points=False, save_path=None, colorbars=False):
    if type(annotation)==xradct.PointAnnotationSet:
        ann_points = annotation.points
        ann_labels = annotation.labels
    elif type(annotation)==list:
        ann_points = annotation
    center_i = voxels.shape[0]//2
    center_j = voxels.shape[1]//2
    center_k = voxels.shape[2]//2

    default_text_color = plt.rcParams['text.color']
    plt.rcParams.update({'text.color': "white"})
    if not save_path==None:
        fig = plt.figure(figsize=[11, 3], dpi=600)
    else:
        fig = plt.figure(figsize=[18,7])

    (x_disp, y_disp, z_disp) =  fig.subplots(1, 3, gridspec_kw={'width_ratios': [voxels.shape[1], voxels.shape[0], voxels.shape[0]]})
    if CT_scale:
        imx = x_disp.imshow((voxels[center_i,:,:].T), vmin=-800, vmax=1800, cmap=cmap)
        x_disp.invert_yaxis()
        x_disp.set_xlabel('y')
        x_disp.set_ylabel('z')
        imy = y_disp.imshow((voxels[:,center_j,:].T), vmin=-800, vmax=1800, cmap=cmap)
        y_disp.invert_yaxis()
        y_disp.set_xlabel('x')
        y_disp.set_ylabel('z')
        imz = z_disp.imshow((voxels[:,:,center_k].T), vmin=-800, vmax=1800, cmap=cmap)
        z_disp.invert_yaxis()
        z_disp.set_xlabel('x')
        z_disp.set_ylabel('y')
        if colorbars:
            plt.colorbar(imx, ax=x_disp)
            plt.colorbar(imy, ax=y_disp)
            plt.colorbar(imz, ax=z_disp)
        for point in annotation:
            x_disp.scatter(point[1], point[2])
            y_disp.scatter(point[0], point[2])
            z_disp.scatter(point[0], point[1])
        for point, label in zip(ann_points, ann_labels):
            x_disp.annotate(label, (point[1], point[2]), color='#FEFEFE')
            y_disp.annotate(label, (point[0], point[2]), color='#FEFEFE')
            z_disp.annotate(label, (point[0], point[1]), color='#FEFEFE')
        plt.show()
    else:
        imx = x_disp.imshow((voxels[center_i,:,:].T), cmap=cmap)
        x_disp.invert_yaxis()
        x_disp.set_xlabel('y')
        x_disp.set_ylabel('z')
        #plt.colorbar(imx, ax=x_disp)
        imy = y_disp.imshow((voxels[:,center_j,:].T), cmap=cmap)
        y_disp.invert_yaxis()
        y_disp.set_xlabel('x')
        y_disp.set_ylabel('z')
        #plt.colorbar(imy, ax=y_disp)
        imz = z_disp.imshow((voxels[:,:,center_k].T), cmap=cmap)
        z_disp.invert_yaxis()
        z_disp.set_xlabel('x')
        z_disp.set_ylabel('y')
        #plt.colorbar(imz, ax=z_disp)
        if colorbars:
            plt.colorbar(imx, ax=x_disp)
            plt.colorbar(imy, ax=y_disp)
            plt.colorbar(imz, ax=z_disp)
        for point in ann_points:
            x_disp.scatter(point[1], point[2])
            y_disp.scatter(point[0], point[2])
            z_disp.scatter(point[0], point[1])
        if label_points:
            for point, label in zip(ann_points, ann_labels):
                x_disp.annotate(label, (point[1], point[2]), color='#FEFEFE')
                y_disp.annotate(label, (point[0], point[2]), color='#FEFEFE')
                z_disp.annotate(label, (point[0], point[1]), color='#FEFEFE')


        if not save_path==None:
            fig.tight_layout()
            fig.savefig(save_path, bbox_inches='tight')

        plt.show()
    plt.rcParams.update({'text.color': default_text_color})

# Unit Tests
def test_of_image_display_functions():
    test_scan_path = r'S:\RADONC\Karam_Lab\AutoContour\Test Data\One Mouse\A4'
    print(f'reading DICOM files from {test_scan_path}')
    voxels = voxels_from_dicom_folder(test_scan_path)[0]

    view_center_slices(voxels)
    view_xyz_means(voxels)
    view_xyz_mips(voxels)

def test_of_translate_with_point_pairs():
    first_point = (50, 50, 101)
    second_point = (70, 90, 102)
    third_point = (80, 30, 103)
    fourth_point = (150, 80, 104)
    point_pairs = [(first_point, second_point), (third_point, fourth_point)]
    points_list = [first_point, second_point, third_point, fourth_point]

    voxels = np.zeros([200,200,200])
    x, y, z = index_grid(voxels)

    #x_out, y_out, z_out = translate_with_point_pairs(np.array([point[0] for point in points_list]),
    #                                                 np.array([point[1] for point in points_list]),
    #                                                 np.array([point[2] for point in points_list]),
    #                                                 point_pairs)
    x_out, y_out, z_out, right = translate_with_point_pairs(x, y, z, point_pairs)
    view_center_slices((np.sign(x-x_out)+np.sign(y-y_out)+np.sign(z-z_out)), annotation=points_list, CT_scale=False)
    view_center_slices(right, annotation=points_list, CT_scale=False)


if __name__ == '__main__':
    test_of_image_display_functions()
