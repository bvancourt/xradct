import numpy as np
import cc3d
import importlib
import scipy.signal
import matplotlib.pyplot as plt
from copy import deepcopy
import os

import voxelhelp
import xradct
import scan_processing
import vector_fields_3D as vf3d

importlib.reload(vf3d)
importlib.reload(voxelhelp)
importlib.reload(xradct)
importlib.reload(scan_processing)

def final_head(mouse_only_scan, save_seg_file=False, seg_file_path='./head_seg.nii', save_scan_file=False, scan_file_path='./head_scan.nii', force_use_head_only=False):

    headscan_seg, split_scan_seg, headscan, segmentation_info = segment_head(mouse_only_scan, force_use_head_only=force_use_head_only)

    segmentation_info['error_strings'] = []

    segmentation_info['all_warning_strings'] = (
        segmentation_info['tooth_seg_info']['warning_strings'] +
        segmentation_info['bone_seg_info']['warning_strings'] +
        segmentation_info['resample_info']['warning_strings'] + 
        segmentation_info['warning_strings']
    )

    if np.sum(headscan_seg.int_image)==0:
        segmentation_info['error_strings'].append('final_head() generated empty segmentation image.')

    if save_seg_file:
        if seg_file_path=='./head_seg.nii': # If file name is default, add the mouse ID before saving.
            seg_file_path = f'./{mouse_only_scan.name}_head_seg.nii'

        voxelhelp.write_nii(headscan_seg.int_image, seg_file_path)

    if save_scan_file:
        if scan_file_path=='./head_scan.nii': # If file name is default, add the mouse ID before saving.
            scan_file_path = f'./{mouse_only_scan.name}_head_scan.nii'

        voxelhelp.write_nii(headscan.voxels, scan_file_path)

    return headscan_seg, split_scan_seg, headscan, segmentation_info

def segment_teeth(scan):
    seg = xradct.Segmentation.empty(scan.voxels.shape)
    points = xradct.PointAnnotationSet.empty()
    tooth_seg_info = {
        'warning_strings' : []
    }

    teeth_target_volume = 4800
    bone_thresh = 600
    counts, bins = np.histogram(scan.voxels, bins=500)
    bin_means = (bins[:-1]+bins[1:])/2
    teeth_thresh = np.max(np.flip(bin_means)[np.cumsum(np.flip(counts))>4800])
    tooth_seg_info['teeth_thresh'] = teeth_thresh
    #teeth_mask = voxelhelp.exclude_small_connected(
    #    (voxelhelp.mask_erode(scan.voxels>teeth_thresh,1)), 200, connectivity=6,
    #)&(scan.y>scan.voxels.shape[1]-150)
    teeth_mask = voxelhelp.mask_dilate(voxelhelp.exclude_small_connected((scan.voxels>teeth_thresh), 100, connectivity=6), 2)&(scan.voxels>bone_thresh)

    #voxelhelp.view_xyz_mips((1+4*teeth_mask)*scan.voxels)
    
    teeth_point = voxelhelp.mask_coordinate_centroid(teeth_mask, scan.x, scan.y, scan.z)
    bone_near_teeth = (scan.voxels>bone_thresh)&voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 50, teeth_point)
    teeth_mask &= bone_near_teeth
    head_point = voxelhelp.mask_coordinate_centroid(bone_near_teeth, scan.x, scan.y, scan.z)
    
    # Isolate front and bottom teeth.
    back_teeth = teeth_mask&voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 20, head_point)
    bottom_teeth = voxelhelp.largest_connected_only(teeth_mask&~back_teeth, connectivity=6)
    front_teeth = voxelhelp.second_largest_connected_only(teeth_mask&~back_teeth, connectivity=6)

    if np.sum(bottom_teeth)>3*np.sum(front_teeth): # this would likely be caused by the teeth_thresh being too low and the teeth all getting connected
        tooth_seg_info['warning_strings'].append('Warning!! Teeth may not have been accurately separated!')
    if np.mean(scan.y[bottom_teeth])>np.mean(scan.y[front_teeth]):
        bottom_teeth, front_teeth = front_teeth, bottom_teeth

    
        
    # Find the point where front teeth come closest to touching bottom teeth.
    proximity_kernel_radius = 22
    proximity_kernel = voxelhelp.inverse_square_kernel(proximity_kernel_radius)*voxelhelp.spherical_kernel(proximity_kernel_radius)
    bite_point = np.unravel_index(np.argmax(
        scipy.signal.fftconvolve(bottom_teeth, proximity_kernel, mode='same')*
        scipy.signal.fftconvolve(front_teeth, proximity_kernel, mode='same')
    ), scan.voxels.shape)

    # Define bottom and front teeth regions and centroids. This is basically 2 iterations of finding a centroid, then restricting the region to a fixed distance from it.
    front_teeth_centroid = voxelhelp.mask_coordinate_centroid(
        front_teeth&voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 25, bite_point), scan.x, scan.y, scan.z)
    bottom_teeth_centroid = voxelhelp.mask_coordinate_centroid(
        bottom_teeth&voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 25, bite_point), scan.x, scan.y, scan.z)
    front_teeth &= voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 15, front_teeth_centroid)
    #print(f'front_teeth volume 1: {np.sum(front_teeth)}')
    bottom_teeth &= voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 15, bottom_teeth_centroid)
    
    points.add_point(voxelhelp.mask_coordinate_centroid(front_teeth, scan.x, scan.y, scan.z), 'Front Teeth')
    points.add_point(voxelhelp.mask_coordinate_centroid(bottom_teeth, scan.x, scan.y, scan.z), 'Bottom Teeth')
    points.add_point(bite_point, 'Bite Point')
    #points.add_point(head_point, 'Head Point')

    # Define back teeth region
    approx_normal = voxelhelp.three_points_to_normal(head_point, front_teeth_centroid, bottom_teeth_centroid)
    approx_right = (scan.x-head_point[0])*approx_normal[0] + (scan.y-head_point[1])*approx_normal[1] + (scan.z-head_point[2])*approx_normal[2] > 0

    #voxelhelp.view_xyz_mips(scan.voxels*approx_right*back_teeth)
    right_back_teeth_centroid = voxelhelp.mask_coordinate_centroid(approx_right&back_teeth, scan.x, scan.y, scan.z)
    left_back_teeth_centroid = voxelhelp.mask_coordinate_centroid((~approx_right)&back_teeth, scan.x, scan.y, scan.z)
    back_teeth_point = voxelhelp.tuple_mean([right_back_teeth_centroid, left_back_teeth_centroid]) # this should be closer to center than the head_point

    back_teeth = teeth_mask&voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 17, back_teeth_point) # iterating to imporve centering
    right_back_teeth_centroid = voxelhelp.mask_coordinate_centroid(approx_right&back_teeth, scan.x, scan.y, scan.z)
    left_back_teeth_centroid = voxelhelp.mask_coordinate_centroid((~approx_right)&back_teeth, scan.x, scan.y, scan.z)
    back_teeth_point = voxelhelp.tuple_mean([right_back_teeth_centroid, left_back_teeth_centroid])
    back_teeth = teeth_mask&(voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 6, right_back_teeth_centroid)|
                             voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 6, left_back_teeth_centroid))
    points.add_point(back_teeth_point, 'Back Teeth')

    approx_normal = voxelhelp.three_points_to_normal(back_teeth_point, front_teeth_centroid, bottom_teeth_centroid)
    approx_right = (scan.x-back_teeth_point[0])*approx_normal[0] + (scan.y-back_teeth_point[1])*approx_normal[1] + (scan.z-back_teeth_point[2])*approx_normal[2] > 0

    #mirror_x, mirror_y, mirror_z = voxelhelp.mirror_coordinates(x, y, z, back_teeth_point, approx_normal)
    
    #seg.add_label(teeth_mask&~back_teeth, 'Rough Teeth Mask')
    #seg.add_label(voxelhelp.mask_dilate(back_teeth, 2)&(scan.voxels>bone_thresh), 'Back Teeth')
    #seg.add_label(voxelhelp.mask_dilate(front_teeth, 3)&(scan.voxels>bone_thresh)&~seg['Back Teeth'], 'Front Teeth')
    #seg.add_label(voxelhelp.mask_dilate(bottom_teeth, 3)&(scan.voxels>bone_thresh)&~seg['Back Teeth'], 'Bottom Teeth')

    seg.add_label(voxelhelp.mask_dilate(back_teeth, 2)&(scan.voxels>bone_thresh)&approx_right, 'Right Back Teeth')
    seg.add_label(voxelhelp.mask_dilate(back_teeth, 2)&(scan.voxels>bone_thresh)&(~approx_right), 'Left Back Teeth')
    seg.new_union('Back Teeth', ['Right Back Teeth', 'Left Back Teeth'])
    seg.add_label(voxelhelp.mask_dilate(bottom_teeth, 3)&(scan.voxels>bone_thresh)&approx_right, 'Right Bottom Tooth')
    seg.add_label(voxelhelp.mask_dilate(bottom_teeth, 3)&(scan.voxels>bone_thresh)&(~approx_right), 'Left Bottom Tooth')
    seg.new_union('Bottom Teeth', ['Right Bottom Tooth', 'Left Bottom Tooth'])
    seg.add_label(voxelhelp.mask_dilate(front_teeth, 3)&(scan.voxels>bone_thresh)&approx_right, 'Right Front Tooth')
    seg.add_label(voxelhelp.mask_dilate(front_teeth, 3)&(scan.voxels>bone_thresh)&(~approx_right), 'Left Front Tooth')
    if (np.sum(seg['Right Front Tooth']|seg['Left Front Tooth'])==0) and not (np.sum(front_teeth)==0):
        tooth_seg_info['warning_strings'].append('Left and right front teeth labels were empty; falling back to non-union "Front Teeth" label.')
        seg.add_label(front_teeth, 'Front Teeth')
    else:
        seg.new_union('Front Teeth', ['Right Front Tooth', 'Left Front Tooth'])

    return seg, points, tooth_seg_info

def autothresh_skeleton(scan, teeth_points, plot_bone_thresh_hist=False):
    tissue_thresh = -150
    bone_thresh_quantile = .73
    rough_bone_thresh = 550
    autothresh_confidence = .85

    # Calculate bone threshold, since it likely needs to be lower for blurrier scans
    
    teeth_exclusion_radius = voxelhelp.dist_a_to_b(teeth_points['Back Teeth'], teeth_points['Bite Point'])*1.3
    bone_test_region = ((scan.voxels>tissue_thresh)
        &voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, teeth_exclusion_radius+31, teeth_points['Bite Point'])
        &~voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, teeth_exclusion_radius, teeth_points['Bite Point'])
        )
    bone_test_region &= voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 31,
        voxelhelp.mask_coordinate_centroid(bone_test_region&(scan.voxels > rough_bone_thresh), scan.x, scan.y, scan.z)
        )
    bone_thresh = np.quantile(scan.voxels[bone_test_region], bone_thresh_quantile)*autothresh_confidence + rough_bone_thresh*(1-autothresh_confidence)

    if plot_bone_thresh_hist:
        print(f'Bone threshold: {bone_thresh}')
        counts, bins = np.histogram(scan.voxels[bone_test_region], bins=500)
        bin_means = (bins[:-1]+bins[1:])/2
        smoothing_kernel = scipy.signal.convolve(np.ones(6),np.ones(6))
        smoothing_kernel /= np.sum(smoothing_kernel)

        smoothed_counts = scipy.signal.convolve(counts, smoothing_kernel, mode='same')

        plt.plot(bin_means, smoothed_counts)
        plt.plot([bone_thresh, bone_thresh], [np.min(smoothed_counts), np.max(smoothed_counts)])
        plt.show()

    return (scan.voxels>bone_thresh), bone_thresh

def isolate_spine_and_arms(skeleton_mask, scan):
    # make filter kernels
    center_cut_depth = .75
    spine_filter_range = 9
    spine_filter_param = np.arange(-spine_filter_range, spine_filter_range+1)
    sf_x, sf_y, sf_z = np.meshgrid(spine_filter_param, spine_filter_param, spine_filter_param)
    sf_r = np.sqrt(sf_x*sf_x + sf_y*sf_y + sf_z*sf_z)
    sf_r *= (sf_r<spine_filter_range)
    spine_filter_kernel = (sf_r**2-sf_r**4/spine_filter_range**2)**3
    spine_filter_kernel /= np.sum(spine_filter_kernel)
    spherical_kernel = voxelhelp.spherical_window(spine_filter_kernel)
    spine_filter_kernel -= center_cut_depth*spherical_kernel/np.sum(spherical_kernel)

    # This set of operations gets a subset of the skeleton that is not really just spine, but at least doesn't include ribs or clavicles.
    spine_filtered_skeleton = scipy.signal.convolve(skeleton_mask, spine_filter_kernel, mode='same')
    sphere_filtered_sfs = skeleton_mask&voxelhelp.mask_erode(scipy.signal.convolve(spine_filtered_skeleton*spine_filtered_skeleton, spherical_kernel, mode='same')>.9, 4)
    #voxelhelp.view_xyz_mips(sphere_filtered_sfs)

    spine_and_skull = skeleton_mask*voxelhelp.mask_dilate(voxelhelp.largest_connected_only(sphere_filtered_sfs, connectivity=6), 3)
    first_arm = skeleton_mask*voxelhelp.mask_dilate(voxelhelp.largest_connected_only(sphere_filtered_sfs&~spine_and_skull, connectivity=6), 8)
    second_arm = skeleton_mask*voxelhelp.mask_dilate(voxelhelp.largest_connected_only(sphere_filtered_sfs&~spine_and_skull&~first_arm, connectivity=6), 8)

    if np.mean(scan.x[first_arm])>np.mean(scan.x[second_arm]):
        return spine_and_skull, first_arm, second_arm
    else:
        return spine_and_skull, second_arm, first_arm

def cranium_reg_points(scan, teeth_seg, teeth_points, bone_seg, bone_points, approx_normal, show_plots=False, use_head_only=False):
    points = xradct.PointAnnotationSet.empty()
    midline_exclusion_distance = 10
    thresh_above_tissue = 200
    warning_strings = []

    bone_only_voxels = np.maximum(thresh_above_tissue, scan.voxels)-thresh_above_tissue

    # Locate cheek bones
    reg_anchor_potential = voxelhelp.reg_anchor_filt(bone_only_voxels, .5, extent_sigma=5, detail_scale_mm=.3)*scan.voxels # tuned for cheek points
    approx_far_right = ((scan.x-teeth_points['Back Teeth'][0])*approx_normal[0] + (scan.y-teeth_points['Back Teeth'][1])*approx_normal[1] 
        + (scan.z-teeth_points['Back Teeth'][2])*approx_normal[2] > midline_exclusion_distance)
    approx_far_left = ((scan.x-teeth_points['Back Teeth'][0])*approx_normal[0] + (scan.y-teeth_points['Back Teeth'][1])*approx_normal[1] 
        + (scan.z-teeth_points['Back Teeth'][2])*approx_normal[2] < -midline_exclusion_distance)
    if not use_head_only:
        spine_side = voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 45, bone_points['Upper Cervical Spine'])
    else:
        spine_side = voxelhelp.closer_to_a_than_b(scan.x, scan.y, scan.z, voxelhelp.tuple_mean([bone_points['Left Jaw Tip'], bone_points['Right Jaw Tip']]), teeth_points['Front Teeth'])
    near_teeth = voxelhelp.mask_dilate(teeth_seg['Front Teeth']|teeth_seg['Back Teeth'], 8)
    allowable_cheek_rgion = (bone_seg['Cranium']&~near_teeth&(approx_far_left|approx_far_right)&~spine_side)
    
    # Registered left to right and righ to left then averaged for maximum accuracy and standardization between scans
    left_cheek_point = np.unravel_index(np.argmax(reg_anchor_potential*(allowable_cheek_rgion&approx_far_left)), scan.voxels.shape)
    right_cheek_point = np.unravel_index(np.argmax(reg_anchor_potential*(allowable_cheek_rgion&approx_far_right)), scan.voxels.shape)
    if np.sum(np.array(left_cheek_point)-np.array(right_cheek_point))<10:
        warning_strings.append('warning!!: cranium_reg_points() found cheeks to be too close together.')

    left_cheek_reg_point, _ = voxelhelp.adjust_point_pair_by_registration(bone_only_voxels, scan.x, scan.y, scan.z, left_cheek_point, right_cheek_point, 30, 8, view_blocks=False)
    right_cheek_reg_point, _ = voxelhelp.adjust_point_pair_by_registration(bone_only_voxels, scan.x, scan.y, scan.z, right_cheek_point, left_cheek_point, 30, 8, view_blocks=False)
    points.add_point(voxelhelp.tuple_mean([left_cheek_point, left_cheek_reg_point]), 'Left Cheek')
    points.add_point(voxelhelp.tuple_mean([right_cheek_point, right_cheek_reg_point]), 'Right Cheek')
    #voxelhelp.view_xyz_mips(reg_anchor_potential*allowable_cheek_rgion, annotation=[left_cheek_point,right_cheek_point,left_cheek_reg_point,right_cheek_reg_point]) # debug image for cheek point location

    # Locate Ear Points
    reg_anchor_potential = voxelhelp.reg_anchor_filt(np.maximum(200, scan.voxels)-200, 1, extent_sigma=3, detail_scale_mm=.5)*scan.voxels # tuned for ear points
    reg_anchor_potential -= np.min(reg_anchor_potential)
    #reg_anchor_potential *= voxelhelp.closer_to_a_than_b(scan.x, scan.y, scan.z, 
    #    voxelhelp.tuple_mean([bone_points['Left Jaw Tip'], bone_points['Right Jaw Tip']]),
    #    voxelhelp.tuple_mean([points['Left Cheek'], points['Right Cheek']]))

    if not use_head_only:
        approx_right = ((scan.x-bone_points['Upper Cervical Spine'][0])*approx_normal[0] + (scan.y-bone_points['Upper Cervical Spine'][1])*approx_normal[1] 
            + (scan.z-bone_points['Upper Cervical Spine'][2])*approx_normal[2] > 0)
    else:
        approx_right = voxelhelp.closer_to_a_than_b(scan.x, scan.y, scan.z, points['Right Cheek'], points['Left Cheek'])

    left_ear_point = np.unravel_index(np.argmax(reg_anchor_potential*(spine_side&~approx_right&~near_teeth&bone_seg['Cranium'])), scan.voxels.shape)
    right_ear_point = np.unravel_index(np.argmax(reg_anchor_potential*(spine_side&approx_right&~near_teeth&bone_seg['Cranium'])), scan.voxels.shape)
    left_ear_reg_point, _ = voxelhelp.adjust_point_pair_by_registration(bone_only_voxels, scan.x, scan.y, scan.z, left_ear_point, right_ear_point, 30, 8, view_blocks=False)
    right_ear_reg_point, _ = voxelhelp.adjust_point_pair_by_registration(bone_only_voxels, scan.x, scan.y, scan.z, right_ear_point, left_ear_point, 30, 8, view_blocks=False)
    points.add_point(voxelhelp.tuple_mean([left_ear_point, left_ear_reg_point]), 'Left Ear')
    points.add_point(voxelhelp.tuple_mean([right_ear_point, right_ear_reg_point]), 'Right Ear')

    if show_plots:
        voxelhelp.view_xyz_mips(reg_anchor_potential*(~approx_right&~near_teeth&bone_seg['Cranium']), annotation=points)
        voxelhelp.view_xyz_mips(scan.voxels*(spine_side), annotation=points)
    
    return points, warning_strings

def segment_bones(scan, teeth_seg, teeth_points, show_plots=False, use_head_only=False):
    seg = xradct.Segmentation.empty(scan.voxels.shape)
    points = xradct.PointAnnotationSet.empty()
    bone_seg_info = {
        'warning_strings': []
    }

    teeth_tips_exclusion = voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 7, teeth_points['Bite Point'])
    skeleton_mask, bone_thresh = autothresh_skeleton(scan, teeth_points)
    skeleton_mask &= ~teeth_tips_exclusion
    bone_seg_info['bone_thresh'] = bone_thresh

    # Region grow from front teeth to separate skull from other bones
    n_skull_segments = 75
    

    front_teeth_safe = teeth_seg['Front Teeth']&~teeth_tips_exclusion
    if np.sum(front_teeth_safe) == 0:
        front_teeth_safe = teeth_seg['Front Teeth']
        if np.sum(front_teeth_safe) == 0:
            bone_seg_info['warning_strings'].append('segment_bones() recieved teeth_seg with empty "Front Teeth" label and tehrefore cannot segemnt bones.')
            assert False==True
        else:
            bone_seg_info['warning_strings'].append('warning!! teeth tips exclusion not used for onion grow from front teeth')
    onion_from_front_teeth, label_sizes = voxelhelp.onion_grow(
        scan.x, scan.y, scan.z,
        front_teeth_safe, skeleton_mask,
        2, n_skull_segments, return_label_sizes=True
    )
    indices = np.arange(len(label_sizes))
    dist_from_front_teeth = voxelhelp.distance_from_point(scan.x, scan.y, scan.z, teeth_points['Front Teeth'])
    label_dist_from_front_teeth = np.array([np.mean(dist_from_front_teeth[onion_from_front_teeth==i]) for i in indices])
    try:
        max_neck_index = np.min(indices[(label_dist_from_front_teeth>120)&(label_sizes<np.max(label_sizes[indices>10])*.25)])
        estimated_neck_loss = 0
    except:
        max_neck_index = np.max(onion_from_front_teeth)
        estimated_neck_loss = 52-max_neck_index # max neck index is expected to be about 52, if the onion grows that far.
        use_head_only = True
        bone_seg_info['warning_strings'].append(f'max_neck_index not detected. Scan likely does not contain enough of the mouse.')
    #if label_dist_from_front_teeth[max_neck_index] > 140:
    #    max_neck_index = np.min(indices[label_dist_from_front_teeth<=140])
    #    print(f'warning!!: max_neck_index reduced because the label was too far from the fron teeth')
    if len(label_sizes) < 62:
        use_head_only = True
    
    skull_and_upper_spine = voxelhelp.largest_connected_only((onion_from_front_teeth<max_neck_index-1*use_head_only)&(onion_from_front_teeth>0))

    if show_plots:
        print('finding max_neck_index from onion_from_front_teeth label sizes.')
        plt.plot(label_sizes)
        plt.plot([np.min(indices), np.max(indices)], [np.max(label_sizes[indices>10])*.25, np.max(label_sizes[indices>10])*.25])
        plt.plot([max_neck_index, max_neck_index], [0, np.max(label_sizes)])
        plt.plot((label_dist_from_front_teeth)*10)
        plt.show()
        
        #voxelhelp.view_xyz_mips(onion_from_front_teeth)

    lower_cervical_spine = voxelhelp.largest_connected_only(voxelhelp.mask_dilate(skull_and_upper_spine, 4)&skeleton_mask&~skull_and_upper_spine)


    points.add_point(voxelhelp.mask_coordinate_centroid(lower_cervical_spine, scan.x, scan.y, scan.z), 'Lower Cervical Spine')
    if show_plots:
        print(f'lower_cervical_spine: {np.sum(lower_cervical_spine)}')
        print(f'skull_and_upper_spine: {np.sum(skull_and_upper_spine)}')
        print(f'skull_and_upper_spine&lower_cervical_spine: {np.sum(skull_and_upper_spine&lower_cervical_spine)}')
    # Region grow from upper spine to define skull and cervical spine
    onion_from_cervical_spine, label_sizes = voxelhelp.onion_grow(
        scan.x, scan.y, scan.z, 
        lower_cervical_spine, skull_and_upper_spine, 
        2, n_skull_segments+15, return_label_sizes=True
    )
    if len(label_sizes) < 25:
        bone_seg_info['warning_strings'].append(f'onion_from_cervical_spine may not have grown. np.sum(onion_from_cervical_spine>1) = {np.sum(onion_from_cervical_spine>1)}')
        lower_cervical_spine = voxelhelp.largest_connected_only(voxelhelp.mask_dilate(skull_and_upper_spine, 4)&skeleton_mask&~skull_and_upper_spine)
        skull_and_upper_spine = voxelhelp.largest_connected_only((onion_from_front_teeth<max_neck_index-2)&(onion_from_front_teeth>0))
        lower_cervical_spine = voxelhelp.largest_connected_only(voxelhelp.mask_dilate(skull_and_upper_spine, 4)&skeleton_mask&~skull_and_upper_spine)
    try:
        test_signal = np.array(label_sizes[1:])*np.diff(np.array(label_sizes)*(1-np.exp(-np.arange(estimated_neck_loss, len(label_sizes)+estimated_neck_loss)/6))/np.arange(1+estimated_neck_loss, 1+len(label_sizes)+estimated_neck_loss))
        test_indices = np.arange(len(test_signal))
        min_skull_index = np.argmax(test_signal*(test_indices>15-estimated_neck_loss)*(test_indices<25-estimated_neck_loss/2))
        if show_plots:
            print('Test signal to locate min_skull_index (index of onion_from_cervical_spine beyond which all layers are considered skull as opposed to spine).')
            plt.plot(test_signal)
            #plt.plot(label_sizes)
            plt.plot([min_skull_index, min_skull_index], [0, np.max(test_signal)])
            plt.show()
            voxelhelp.view_xyz_mips(onion_from_cervical_spine)
    except:
        min_skull_index = 1
        use_head_only = True
        bone_seg_info['warning_strings'].append('min_skull_index could not be determined form label sizes and was set to a default value of zero.')

    seg.add_label(onion_from_cervical_spine>min_skull_index, 'Skull')
    seg.add_label((onion_from_cervical_spine<=min_skull_index)&(onion_from_cervical_spine>0), 'Cervical Spine')
    upper_cervical_spine = (onion_from_cervical_spine<=min_skull_index)&(onion_from_cervical_spine>min_skull_index-4)
    points.add_point(voxelhelp.mask_coordinate_centroid(upper_cervical_spine, scan.x, scan.y, scan.z), 'Upper Cervical Spine')
    mid_cervical_spine = (onion_from_cervical_spine<=min_skull_index/2+2)&(onion_from_cervical_spine>min_skull_index/2-2)
    points.add_point(voxelhelp.mask_coordinate_centroid(mid_cervical_spine, scan.x, scan.y, scan.z), 'Mid Cervical Spine')

    # Isolation of arms and thoracic spine using isolate_spine_and_arms() filtering method
    if teeth_points['Back Teeth'][1]>140: # This should be true for scans including the lungs.
        spine_and_skull, left_arm, right_arm = isolate_spine_and_arms(skeleton_mask&~seg['Skull'], scan)
    else: # In this case there maye be a small enough amount of spine in the scan that it is identified as an arm.
        spine_and_skull, left_arm, right_arm = isolate_spine_and_arms(skeleton_mask, scan)
    seg.add_label(spine_and_skull&~seg['Skull']&~seg['Cervical Spine'], 'Thoracic Spine')
    seg.add_label(left_arm, 'Left Arm')
    seg.add_label(right_arm, 'Right Arm')


    # Isolation of jaw by region growing
    back_teeth_exclusion = voxelhelp.mask_dilate(teeth_seg['Back Teeth'], 5)
    onion_from_bottom_teeth, label_sizes = voxelhelp.onion_grow(
        scan.x, scan.y, scan.z, 
        teeth_seg['Bottom Teeth']&(~teeth_tips_exclusion), seg['Skull']&~back_teeth_exclusion, 
        2, n_skull_segments, return_label_sizes=True
    )
    try:
        max_jaw_index = np.maximum(15, np.quantile(onion_from_bottom_teeth[seg['Skull']], .28))
    except:
        max_jaw_index = np.minimum(16, len(label_sizes)-1)
        bone_seg_info['warning_strings'].append(f'max_jaw_index was set to {max_jaw_index} by fallback methode, becuase quantile did not work. This is a bad sign for jaw segmentation accuracy.')

    bottom_teeth_onion_indices = np.arange(len(label_sizes)-1)
    try:
        # there should be a peak in the label sizes at the ideal jaw index (where the onion starts filling the cranium). 
        # The specified quantile (.28) tends to be a little low, and this is a way to nuge the max_jaw_index up if it is below the peak.
        candidate_jaw_index = np.min(bottom_teeth_onion_indices[(bottom_teeth_onion_indices>=max_jaw_index)&(label_sizes[1:]<=label_sizes[:-1])])
        if (candidate_jaw_index - max_jaw_index) < 4:
            max_jaw_index = candidate_jaw_index
    except:
        pass


    if show_plots:
        print('onion_from_bottom_teeth label sizes with quantile cutoff for jaw mask')
        plt.plot(label_sizes)
        plt.plot([max_jaw_index, max_jaw_index], [0, np.max(label_sizes)])
        plt.show()
        voxelhelp.view_xyz_mips((((onion_from_bottom_teeth<max_jaw_index)&(onion_from_bottom_teeth>0))*4+1)*scan.voxels, annotation=teeth_points)

    approx_normal = voxelhelp.three_points_to_normal(teeth_points['Back Teeth'], teeth_points['Front Teeth'], teeth_points['Bottom Teeth'])
    approx_right = (scan.x-teeth_points['Back Teeth'][0])*approx_normal[0] + (scan.y-teeth_points['Back Teeth'][1])*approx_normal[1] + (scan.z-teeth_points['Back Teeth'][2])*approx_normal[2] > 0

    # Back teeth region is split between jaw and cranium based on being above or below the back teeth point, so we need an approximate up vector.
    if not use_head_only:
        approx_up = voxelhelp.three_points_to_normal(
            points['Upper Cervical Spine'],
            voxelhelp.mask_coordinate_centroid(teeth_seg['Right Back Teeth'], scan.x, scan.y, scan.z),
            voxelhelp.mask_coordinate_centroid(teeth_seg['Left Back Teeth'], scan.x, scan.y, scan.z)
        )
    else:
        approx_up = voxelhelp.three_points_to_normal(
            teeth_points['Bite Point'],
            voxelhelp.mask_coordinate_centroid(teeth_seg['Back Teeth']&~approx_right, scan.x, scan.y, scan.z),
            voxelhelp.mask_coordinate_centroid(teeth_seg['Back Teeth']&approx_right, scan.x, scan.y, scan.z)
        )
    approx_lower_split = (scan.x-teeth_points['Back Teeth'][0])*approx_up[0] + (scan.y-teeth_points['Back Teeth'][1])*approx_up[1] + (scan.z-teeth_points['Back Teeth'][2])*approx_up[2] > 0

    jaw_mask = ((onion_from_bottom_teeth<max_jaw_index)&(onion_from_bottom_teeth>0))|(back_teeth_exclusion&approx_lower_split)
    jaw_mask |= seg['Skull']&voxelhelp.mask_dilate(jaw_mask, 4)&~voxelhelp.largest_connected_only(seg['Skull']&~jaw_mask) # recapture jaw tips, which are sometimes lost by the quantile thing.
    seg.subdivide_label('Skull', ['Cranium', 'Jaw'], jaw_mask)

    # Locate jaw tip points (to be used to calculate jaw normal vector)
    

    lower_right_jaw = seg['Jaw']*approx_right&approx_lower_split
    lower_left_jaw = seg['Jaw']*(~approx_right)&approx_lower_split

    jaw_tipness = onion_from_bottom_teeth*(onion_from_cervical_spine-min_skull_index)*onion_from_front_teeth
    #voxelhelp.view_xyz_mips(((jaw_tipness>np.quantile(jaw_tipness[lower_right_jaw], .97))*1+1)*lower_right_jaw) # Debug image for jaw tip region

    points.add_point(voxelhelp.mask_coordinate_centroid((jaw_tipness>np.quantile(jaw_tipness[lower_right_jaw], .97))&lower_right_jaw, scan.x, scan.y, scan.z), 'Right Jaw Tip')
    points.add_point(voxelhelp.mask_coordinate_centroid((jaw_tipness>np.quantile(jaw_tipness[lower_left_jaw], .97))&lower_left_jaw, scan.x, scan.y, scan.z), 'Left Jaw Tip')

    if show_plots:
        print('Jaw Tipness')
        voxelhelp.view_xyz_mips(jaw_tipness)
        voxelhelp.view_xyz_mips(jaw_tipness*(lower_right_jaw*1+lower_left_jaw*1))

    cranium_points, warning_strings = cranium_reg_points(scan, teeth_seg, teeth_points, seg, points, approx_normal, show_plots=show_plots, use_head_only=use_head_only)
    points += cranium_points # absorbs other point set.
    bone_seg_info['warning_strings'] += warning_strings

    # locate shoulder points
    reg_anchor_potential = voxelhelp.reg_anchor_filt(np.maximum(200, scan.voxels)-200, 2, extent_sigma=3, detail_scale_mm=.5)
    near_cervical_spine = voxelhelp.coordinate_sphere_mask(scan.x, scan.y, scan.z, 48, points['Mid Cervical Spine']) # needed to exclude elbows
    points.add_point(np.unravel_index(np.argmax(reg_anchor_potential*(seg['Left Arm']&near_cervical_spine)), scan.voxels.shape), 'Left Shoulder')
    points.add_point(np.unravel_index(np.argmax(reg_anchor_potential*(seg['Right Arm']&near_cervical_spine)), scan.voxels.shape), 'Right Shoulder')

    #voxelhelp.view_xyz_mips(((near_cervical_spine)*4+1)*scan.voxels) # Debug image for near_cervical_spine
    bone_seg_info['use_head_only'] = use_head_only
    return seg, points, bone_seg_info

def make_headscan(scan, teeth_seg, teeth_points, bone_seg, bone_points, show_plots=False, use_head_only=False, upscale_factor=2, fig_save_base_path=None):
    resample_info = {
        'warning_strings' : []
    }

    points = xradct.PointAnnotationSet.empty()

    # Define additional points for coordiante mapping
    points.add_point(voxelhelp.tuple_mean([bone_points['Left Cheek'], bone_points['Right Cheek']]), 'Cheek Center')
    points.add_point(voxelhelp.tuple_mean([bone_points['Left Ear'], bone_points['Right Ear']]), 'Ear Center')
    points.add_point(voxelhelp.tuple_mean([bone_points['Left Jaw Tip'], bone_points['Right Jaw Tip']]), 'Jaw Tip Center')
    points.add_point(voxelhelp.tuple_mean([bone_points['Left Cheek'], bone_points['Left Ear']]), 'Left Mid Cranium')
    points.add_point(voxelhelp.tuple_mean([bone_points['Right Cheek'], bone_points['Right Ear']]), 'Right Mid Cranium')
    points.add_point(voxelhelp.tuple_mean([points['Left Mid Cranium'], points['Right Mid Cranium']]), 'Mid Cranium')
    points.add_point(voxelhelp.tuple_mean([bone_points['Left Shoulder'], bone_points['Right Shoulder']]), 'Shoulder Center')


    # define diff directions at various locations from point pairs
    cranium_normal = -(np.array(points['Right Mid Cranium']) - np.array(points['Left Mid Cranium']))
    cranium_normal /= np.sqrt(np.sum(cranium_normal*cranium_normal))

    cranium_forward = np.array(points['Cheek Center']) - np.array(points['Ear Center'])
    cranium_forward /= np.sqrt(np.sum(cranium_forward*cranium_forward))

    cheek_normal = -(np.array(bone_points['Right Cheek']) - np.array(bone_points['Left Cheek']))
    cheek_normal /= np.sqrt(np.sum(cheek_normal*cheek_normal))

    ear_normal = -(np.array(bone_points['Right Ear']) - np.array(bone_points['Left Ear']))
    ear_normal /= np.sqrt(np.sum(ear_normal*ear_normal))

    jaw_tip_normal = -(np.array(bone_points['Right Jaw Tip']) - np.array(bone_points['Left Jaw Tip']))
    jaw_tip_normal /= np.sqrt(np.sum(jaw_tip_normal*jaw_tip_normal))

    left_shoulder_normal = np.array(bone_points['Left Shoulder']) - np.array(points['Shoulder Center'])
    left_shoulder_normal /= np.sqrt(np.sum(left_shoulder_normal*left_shoulder_normal))

    right_shoulder_normal = np.array(points['Shoulder Center']) - np.array(bone_points['Right Shoulder'])
    right_shoulder_normal /= np.sqrt(np.sum(right_shoulder_normal*right_shoulder_normal))

    jaw_not_normal = np.array(teeth_points['Bottom Teeth']) - np.array(points['Jaw Tip Center'])
    jaw_not_normal /= np.sqrt(np.sum(jaw_not_normal*jaw_not_normal))

    throat_not_normal = np.array(points['Jaw Tip Center']) - np.array(points['Ear Center'])
    throat_not_normal /= np.sqrt(np.sum(throat_not_normal*throat_not_normal))

    front_teeth_not_normal = np.array(teeth_points['Front Teeth']) - np.array(points['Cheek Center'])
    front_teeth_not_normal /= np.sqrt(np.sum(front_teeth_not_normal*front_teeth_not_normal))

    upper_cervical_not_normal = np.array(points['Ear Center']) - np.array(bone_points['Mid Cervical Spine'])
    upper_cervical_not_normal /= np.sqrt(np.sum(upper_cervical_not_normal*upper_cervical_not_normal))

    lower_cervical_not_normal = np.array(bone_points['Mid Cervical Spine']) - np.array(bone_points['Lower Cervical Spine'])
    lower_cervical_not_normal /= np.sqrt(np.sum(lower_cervical_not_normal*lower_cervical_not_normal))

    mid_shoulder_not_normal = np.array(bone_points['Mid Cervical Spine']) - np.array(bone_points['Lower Cervical Spine'])


    # additional vectors derived from the above
    cranium_up = -np.cross(cranium_forward, cranium_normal)
    cranium_up /= np.sqrt(np.sum(cranium_up*cranium_up))

    cheek_up = -np.cross(cranium_forward, cheek_normal)
    cheek_up /= np.sqrt(np.sum(cheek_up*cheek_up))
    cheek_forward = -np.cross(cheek_normal, cheek_up)
    cheek_forward /= np.sqrt(np.sum(cheek_forward*cheek_forward))

    ear_up = -np.cross(cranium_forward, ear_normal)
    ear_up /= np.sqrt(np.sum(ear_up*ear_up))
    ear_forward = -np.cross(ear_normal, ear_up)
    ear_forward /= np.sqrt(np.sum(ear_forward*ear_forward))

    front_teeth_normal = -np.cross(cranium_up, front_teeth_not_normal)
    front_teeth_normal /= np.sqrt(np.sum(front_teeth_normal*front_teeth_normal))
    front_teeth_forward = -np.cross(front_teeth_normal, cranium_up)
    front_teeth_forward /= np.sqrt(np.sum(front_teeth_forward*front_teeth_forward))

    mid_shoulder_normal = left_shoulder_normal+right_shoulder_normal
    mid_shoulder_normal /= np.sqrt(np.sum(mid_shoulder_normal*mid_shoulder_normal))
    mid_shoulder_not_normal = np.cross(lower_cervical_not_normal, mid_shoulder_normal)
    mid_shoulder_not_normal /= np.sqrt(np.sum(mid_shoulder_not_normal*mid_shoulder_not_normal))

    mid_shoulder_not_normal /= np.sqrt(np.sum(mid_shoulder_not_normal*mid_shoulder_not_normal))

    #shoulder_up = -mid_shoulder_not_normal+lower_cervical_not_normal
    shoulder_up = cranium_up
    shoulder_up /= np.sqrt(np.sum(shoulder_up*shoulder_up))

    left_shoulder_forward = -np.cross(left_shoulder_normal, shoulder_up)
    left_shoulder_forward /= np.sqrt(np.sum(left_shoulder_forward*left_shoulder_forward))

    left_shoulder_up = np.cross(left_shoulder_normal, left_shoulder_forward)
    left_shoulder_up /= np.sqrt(np.sum(left_shoulder_up*left_shoulder_up))

    right_shoulder_forward = -np.cross(right_shoulder_normal, shoulder_up)
    right_shoulder_forward /= np.sqrt(np.sum(right_shoulder_forward*right_shoulder_forward))

    right_shoulder_up = np.cross(right_shoulder_normal, right_shoulder_forward)
    right_shoulder_up /= np.sqrt(np.sum(right_shoulder_up*right_shoulder_up))

    upper_cervical_up = 3*cranium_up + shoulder_up
    upper_cervical_up /= np.sqrt(np.sum(upper_cervical_up*upper_cervical_up))

    upper_cervical_normal = np.cross(upper_cervical_not_normal, upper_cervical_up)
    upper_cervical_normal /= np.sqrt(np.sum(upper_cervical_normal*upper_cervical_normal))

    upper_cervical_forward = -np.cross(upper_cervical_normal, upper_cervical_up)

    lower_cervical_up = cranium_up + 2*shoulder_up
    lower_cervical_up /= np.sqrt(np.sum(lower_cervical_up*lower_cervical_up))

    lower_cervical_normal = np.cross(lower_cervical_not_normal, lower_cervical_up)
    lower_cervical_normal /= np.sqrt(np.sum(lower_cervical_normal*lower_cervical_normal))
    lower_cervical_forward = -np.cross(lower_cervical_normal, lower_cervical_up)

    mid_shoulder_forward = left_shoulder_forward+right_shoulder_forward+lower_cervical_forward
    mid_shoulder_forward /= np.sqrt(np.sum(mid_shoulder_forward*mid_shoulder_forward))

    jaw_pseudo_up = np.cross(jaw_not_normal, jaw_tip_normal)
    jaw_forward = jaw_not_normal*np.sum(jaw_not_normal*cranium_forward) + jaw_pseudo_up*np.sum(jaw_pseudo_up*cranium_forward)
    jaw_forward /= np.sqrt(np.sum(jaw_forward*jaw_forward))

    jaw_up = -np.cross(jaw_forward, jaw_tip_normal)
    jaw_up /= np.sqrt(np.sum(jaw_up*jaw_up))

    throat_normal = -np.cross(cranium_up, throat_not_normal)
    throat_normal /= np.sqrt(np.sum(throat_normal*throat_normal))

    throat_forward = -np.cross(throat_normal, cranium_up)
    throat_forward /= np.sqrt(np.sum(throat_forward*throat_forward))

    throat_up = (cranium_up+jaw_up)/2
    throat_up /= np.sqrt(np.sum(throat_up*throat_up))

    


    # Calculate headscan locations of scanspace points and vice versa.
    i_cheek = voxelhelp.dist_a_to_b(bone_points['Right Cheek'], bone_points['Left Cheek'])/2
    j_cheek = voxelhelp.dist_a_to_b(points['Cheek Center'], points['Mid Cranium'])
    headspace_left_cheek = np.array([i_cheek, j_cheek, 0])
    headspace_right_cheek = np.array([-i_cheek, j_cheek, 0])

    j_front_teeth = j_cheek + voxelhelp.dist_a_to_b(teeth_points['Front Teeth'], points['Cheek Center'])*np.sum(front_teeth_forward*front_teeth_not_normal)
    k_front_teeth = voxelhelp.dist_a_to_b(teeth_points['Front Teeth'], points['Cheek Center'])*np.sum(cranium_up*front_teeth_not_normal)
    headspace_front_teeth = np.array([0, j_front_teeth, k_front_teeth])
    scanspace_nose = (np.array(points['Cheek Center']) + np.array(teeth_points['Front Teeth']))/2

    i_ear = voxelhelp.dist_a_to_b(bone_points['Right Ear'], bone_points['Left Ear'])/2
    j_ear = -voxelhelp.dist_a_to_b(points['Ear Center'], points['Mid Cranium'])
    headspace_left_ear = np.array([i_ear, j_ear, 0])
    headspace_right_ear = np.array([-i_ear, j_ear, 0])



    j_mid_cervical = j_ear - voxelhelp.dist_a_to_b(points['Ear Center'], bone_points['Mid Cervical Spine'])*np.sum(upper_cervical_forward*upper_cervical_not_normal)
    k_mid_cervical = -voxelhelp.dist_a_to_b(points['Ear Center'], bone_points['Mid Cervical Spine'])*np.sum(upper_cervical_up*upper_cervical_not_normal)
    headspace_mid_cervical = np.array([0, j_mid_cervical, k_mid_cervical])
    scanspace_upper_cervical_mean = (np.array(points['Ear Center']) + np.array(bone_points['Mid Cervical Spine']))/2

    j_lower_cervical = j_mid_cervical-voxelhelp.dist_a_to_b(bone_points['Mid Cervical Spine'], bone_points['Lower Cervical Spine'])*np.sum(lower_cervical_forward*lower_cervical_not_normal)
    k_lower_cervical = -voxelhelp.dist_a_to_b(bone_points['Mid Cervical Spine'], bone_points['Lower Cervical Spine'])*np.sum(lower_cervical_up*lower_cervical_not_normal)
    headspace_lower_cervical = np.array([0, j_lower_cervical, k_lower_cervical])
    scanspace_lower_cervical_mean = (np.array(bone_points['Lower Cervical Spine']) + np.array(bone_points['Mid Cervical Spine']))/2

    i_shoulder = voxelhelp.dist_a_to_b(bone_points['Left Shoulder'], bone_points['Right Shoulder'])/2
    j_shoulder = j_lower_cervical+voxelhelp.dist_a_to_b(points['Shoulder Center'], bone_points['Lower Cervical Spine'])*np.sum(mid_shoulder_not_normal*lower_cervical_forward)
    k_shoulder = k_lower_cervical+voxelhelp.dist_a_to_b(points['Shoulder Center'], bone_points['Lower Cervical Spine'])*np.sum(mid_shoulder_not_normal*lower_cervical_up)
    headspace_mid_shoulder = np.array([0, j_shoulder, k_shoulder])
    #headspace_mid_shoulder_center = np.array([0, (j_shoulder+j_lower_cervical)/2, (k_shoulder+k_lower_cervical)/2])
    scanspace_mid_shoulder_center = (np.array(points['Shoulder Center']) + np.array(bone_points['Lower Cervical Spine']))/2

    headspace_left_shoulder = np.array([i_shoulder, j_shoulder, k_shoulder])
    headspace_right_shoulder = np.array([-i_shoulder, j_shoulder, k_shoulder])
    scanspace_left_clavicle = (np.array(bone_points['Left Shoulder']) + np.array(points['Shoulder Center']))/2
    scanspace_right_clavicle = (np.array(bone_points['Right Shoulder']) + np.array(points['Shoulder Center']))/2

    j_throat = j_ear + voxelhelp.dist_a_to_b(points['Jaw Tip Center'], points['Ear Center'])*np.sum(throat_forward*throat_not_normal)
    k_throat = voxelhelp.dist_a_to_b(points['Jaw Tip Center'], points['Ear Center'])*np.sum(throat_not_normal*cranium_up)
    headspace_throat = np.array([0, j_throat, k_throat])
    scanspace_throat = (np.array(points['Jaw Tip Center']) + np.array(points['Ear Center']))/2


    i_jaw_tip =  voxelhelp.dist_a_to_b(bone_points['Left Jaw Tip'], bone_points['Right Jaw Tip'])/2
    j_jaw = j_throat + voxelhelp.dist_a_to_b(points['Jaw Tip Center'], teeth_points['Bottom Teeth'])*np.sum(jaw_forward*jaw_not_normal)
    k_jaw = k_throat + voxelhelp.dist_a_to_b(points['Jaw Tip Center'], teeth_points['Bottom Teeth'])*np.sum(jaw_up*jaw_not_normal)
    scanspace_jaw = (np.array(points['Jaw Tip Center']) + np.array(teeth_points['Bottom Teeth']))/2
    scanspace_left_jaw = (np.array(bone_points['Left Jaw Tip']) + np.array(teeth_points['Bottom Teeth']))/2
    scanspace_right_jaw = (np.array(bone_points['Right Jaw Tip']) + np.array(teeth_points['Bottom Teeth']))/2

    exclude_jaw = False
    if voxelhelp.dist_a_to_b(bone_points['Left Jaw Tip'], bone_points['Right Jaw Tip'])<25:
        resample_info['warning_strings'].append('warning!! Jaw ignored, becuase jaw tip points were too close together.')
        exclude_jaw == True

    if (np.sum(jaw_tip_normal*ear_normal) < .7) or (np.sum(jaw_forward*cranium_forward) < .6) or (np.sum(jaw_tip_normal*ear_normal) > 1)|(np.sum(jaw_forward*cranium_forward) > 1):
        resample_info['warning_strings'].append('warning!! Jaw ignored, becuase of implausibly poor alignment with head.')
        exclude_jaw == True

    resample_info['exclude_jaw'] = exclude_jaw


    if not use_head_only:
        if ((i_shoulder>j_jaw+10) and not exclude_jaw)|(j_lower_cervical>j_ear+10):
            resample_info['warning_strings'].append('warning!! Neck and shoulder points ignored because they were too far forward.')
            use_head_only == True

        if voxelhelp.dist_a_to_b(bone_points['Left Shoulder'], bone_points['Right Shoulder'])<25:
            resample_info['warning_strings'].append('warning!! Neck and shoulder points ignored, becuase shoulder points were too close together.')
            use_head_only == True

    

    #print(([headspace_mid_shoulder, headspace_lower_cervical], scanspace_mid_shoulder_center, mid_shoulder_normal, mid_shoulder_forward, shoulder_up))
    unfiltered_pair_bar_coord_sets = [
        # cross bars
        ([headspace_left_cheek, headspace_right_cheek], np.array(points['Cheek Center']), cheek_normal, cheek_forward, cheek_up, 'Cheek Horizontal'),
        ([np.array([-(i_cheek+i_ear)/2, 0, 0]), np.array([(i_cheek+i_ear)/2, 0, 0])], np.array(points['Mid Cranium']), cranium_normal, cranium_forward, cranium_up, 'Cranium Horizontal'),
        ([headspace_left_ear, headspace_right_ear], np.array(points['Ear Center']), ear_normal, ear_forward, ear_up, 'Ear Horizontal'),

        # main axial chain
        ([headspace_front_teeth, np.array([0, j_cheek, 0])], scanspace_nose, front_teeth_normal, front_teeth_forward, cranium_up, 'Front Teeth Center'),
        ([np.array([0, j_cheek, 0]), np.array([0, j_ear, 0])], np.array(points['Mid Cranium']), cranium_normal, cranium_forward, cranium_up, 'Cranium Center'),
        

        # cranial side bars
        ([np.array([(i_cheek+i_ear)/2, j_cheek, 0]), np.array([(i_cheek+i_ear)/2, j_ear, 0])], np.array(points['Left Mid Cranium']), cranium_normal, cranium_forward, cranium_up, 'Left Cranium Side Bar'),
        ([np.array([-(i_cheek+i_ear)/2, j_cheek, 0]), np.array([-(i_cheek+i_ear)/2, j_ear, 0])], np.array(points['Right Mid Cranium']), cranium_normal, cranium_forward, cranium_up, 'Right Cranium Side Bar'),
        ([np.array([(i_cheek+i_ear)/2, j_cheek, 20]), np.array([(i_cheek+i_ear)/2, j_ear, 20])], np.array(points['Left Mid Cranium'])+20*cranium_up, cranium_normal, cranium_forward, cranium_up, 'Left Cranium Upper Bar'),
        ([np.array([-(i_cheek+i_ear)/2, j_cheek, 20]), np.array([-(i_cheek+i_ear)/2, j_ear, 20])], np.array(points['Right Mid Cranium'])+20*cranium_up, cranium_normal, cranium_forward, cranium_up, 'Right Cranium Upper Bar'),
    ]
    if not exclude_jaw:
        unfiltered_pair_bar_coord_sets += [
            # jaw
            ([headspace_throat, np.array([0, j_ear, 0])], scanspace_throat, throat_normal, throat_forward, throat_up, 'Throat Center'),
            ([np.array([i_jaw_tip, j_throat, k_throat]), np.array([-i_jaw_tip, j_throat, k_throat])], np.array(points['Jaw Tip Center']), throat_normal, throat_forward, throat_up, 'Jaw Tip Horizontal'),
            ([headspace_throat, np.array([0, j_jaw, k_jaw])], scanspace_jaw, jaw_tip_normal, jaw_forward, jaw_up, 'Jaw Center'),
            ([np.array([i_jaw_tip, j_throat, k_throat]), np.array([0, j_jaw, k_jaw])], scanspace_left_jaw, jaw_tip_normal, jaw_forward, jaw_up, 'Jaw Left Side Bar'),
            ([np.array([-i_jaw_tip, j_throat, k_throat]), np.array([0, j_jaw, k_jaw])], scanspace_right_jaw, jaw_tip_normal, jaw_forward, jaw_up, 'Jaw Right Side Bar'),
        ]

    if not use_head_only:
        unfiltered_pair_bar_coord_sets += [
            # neck
            ([np.array([0, j_ear, 0]), headspace_mid_cervical], scanspace_upper_cervical_mean, upper_cervical_normal, upper_cervical_forward, upper_cervical_up, 'Upper Neck'),
            ([headspace_mid_cervical, headspace_lower_cervical], scanspace_lower_cervical_mean, lower_cervical_normal, lower_cervical_forward, lower_cervical_up, 'Lower Neck'),

            # shoulders
            ([headspace_mid_shoulder, headspace_lower_cervical], scanspace_mid_shoulder_center, mid_shoulder_normal, mid_shoulder_forward, shoulder_up, 'Shoulder Center'),
            ([headspace_left_shoulder, headspace_mid_shoulder], scanspace_left_clavicle, left_shoulder_normal, left_shoulder_forward, left_shoulder_up, 'Left Shoulder'),
            ([headspace_right_shoulder, headspace_mid_shoulder], scanspace_right_clavicle, right_shoulder_normal, right_shoulder_forward, right_shoulder_up, 'Right Shoulder'),
        ]

    # point bar QC
    def aligned(v1, v2, p_min=.7, p_max=1.01):
        return (p_min<=np.sum(v1*v2)<=p_max)

    pair_bar_coord_sets = []
    for pair_bar in unfiltered_pair_bar_coord_sets:
        if aligned(pair_bar[2], cranium_normal) and aligned(pair_bar[3], cranium_forward) and aligned(pair_bar[4], cranium_up):
            pair_bar_coord_sets.append(pair_bar[:-1])
        else:
            resample_info['warning_strings'].append(f'pair bar {pair_bar[-1]} excluded due to poor alignment with cranium')
            # this is to avoid taking only one shoulder
            if pair_bar[-1] == 'Left Shoulder':
                use_head_only = True
                break
            elif pair_bar[-1] == 'Right Shoulder':
                use_head_only = True
                pair_bar_coord_sets = pair_bar_coord_sets[:-1]

    resample_info['use_head_only'] = use_head_only
    # make headscan
    if use_head_only:
        resample_info['warning_strings'].append('Head limit for segemtation set to ears.')
        head_limit = int(np.ceil(j_ear)) 
        resample_info['how_far_back'] = 'Ear'
    else:
        #print('Head limit for segemtation set to neck.')
        head_limit = int(np.ceil(j_mid_cervical)) 
        resample_info['how_far_back'] = 'Neck'

    

    headscan_x_min, headscan_x_max, headscan_y_min, headscan_y_max, headscan_z_min, headscan_z_max = -75*upscale_factor, 75*upscale_factor, head_limit*upscale_factor, 75*upscale_factor, -75*upscale_factor, 50*upscale_factor
    headscan_j, headscan_i, headscan_k = np.meshgrid(np.arange(headscan_y_min, headscan_y_max), np.arange(headscan_x_min, headscan_x_max), np.arange(headscan_z_min, headscan_z_max))
    headscan = np.zeros(headscan_i.shape)
    resamp_grid = np.zeros([headscan_x_max-headscan_x_min, headscan_y_max-headscan_y_min, headscan_z_max-headscan_z_min, 3])

    # glue together a full coordinate system from those defined by the point pairs
    total_pull = np.zeros(headscan.shape, dtype=np.float64)
    for hs_point_pair, scanspace_origin, this_normal, this_forward, this_up in pair_bar_coord_sets:
        hs_center = (hs_point_pair[0]*upscale_factor + hs_point_pair[1]*upscale_factor)/2
        this_pull = 1/voxelhelp.pill_r_squared(headscan_i, headscan_j, headscan_k, hs_point_pair[0]*upscale_factor, hs_point_pair[1]*upscale_factor, min_val=.5)/upscale_factor
        this_pull = this_pull*this_pull

        this_sampling_grid = (
            ((headscan_i-hs_center[0])/upscale_factor)[:,:,:,np.newaxis]*this_normal[np.newaxis,np.newaxis,np.newaxis,:] +
            ((headscan_j-hs_center[1])/upscale_factor)[:,:,:,np.newaxis]*this_forward[np.newaxis,np.newaxis,np.newaxis,:] +
            ((headscan_k-hs_center[2])/upscale_factor)[:,:,:,np.newaxis]*this_up[np.newaxis,np.newaxis,np.newaxis,:] +
            scanspace_origin[np.newaxis,np.newaxis,np.newaxis,:]
        )

        total_pull += this_pull
        current_weight = this_pull/total_pull

        resamp_grid = this_sampling_grid*current_weight[:,:,:,np.newaxis] + resamp_grid*(1-current_weight[:,:,:,np.newaxis])

    resamp_x, resamp_y, resamp_z = vf3d.unstack_vector_to_xyz(resamp_grid)

    headscan = xradct.CTScan(voxelhelp.resamp_volume(scan.voxels, resamp_x, resamp_y, resamp_z), [.2/upscale_factor, .2/upscale_factor, .2/upscale_factor], scan.name, scan.date)
    headscan.resamp_x, headscan.resamp_y, headscan.resamp_z = resamp_x, resamp_y, resamp_z
    headscan.y, headscan.x, headscan.z = headscan_j, headscan_i, headscan_k
    headscan.voxel_volumes = vf3d.cell_volume(resamp_grid)/125

    if show_plots:
        if fig_save_base_path==None:
            voxelhelp.view_xyz_mips(np.log(total_pull))
            voxelhelp.view_xyz_mips(headscan.voxels)
            voxelhelp.view_center_slices(voxelhelp.test_checker(resamp_x, resamp_y, resamp_z), CT_scale=False)
            voxelhelp.view_xyz_mips(np.clip(headscan.voxel_volumes*125, .8, 1.25))

        else:
            voxelhelp.view_xyz_mips(np.log(total_pull), save_path=os.path.join(fig_save_base_path, 'neon_head_frame.png'))
            voxelhelp.view_xyz_mips(headscan.voxels, save_path=os.path.join(fig_save_base_path, 'headscan_mips.png'))
            voxelhelp.view_center_slices(voxelhelp.test_checker(resamp_x, resamp_y, resamp_z), CT_scale=False, save_path=os.path.join(fig_save_base_path, 'warp_checker.png'))
            voxelhelp.view_xyz_mips(np.clip(headscan.voxel_volumes*1000, .8, 1.25), colorbars=True, save_path=os.path.join(fig_save_base_path, 'voxel_volumes.png'), cmap='jet')
    return headscan, resample_info

def correct_headscan_by_gradient_descent(headscan, source_scan, show_plots=False, fig_save_base_path=None):
    learn_rate = 1.15
    rx, ry, rz = deepcopy(headscan.resamp_x), deepcopy(headscan.resamp_y), deepcopy(headscan.resamp_z)
    k = voxelhelp.gaussian_filter_mm(.6, np.array([.1, .1, .1]), 2)
    k /= np.sum(k)
    not_air_mask = headscan.voxels > -100

    last_loss = 100000000
    for n in range(20):
        if show_plots:
            print(f'Iteration {n}')
        learn_rate *= .975
        
        k = voxelhelp.gaussian_filter_mm(.85 - n*(.3/20), np.array([.1, .1, .1]), 2)
        k /= np.sum(k)
        
        #k = voxelhelp.gaussian_filter_mm((20-n/2)/20, np.array([.1, .1, .1]), 2)
        resampled = np.zeros(headscan.voxels.shape, dtype=np.float32)
        resampled_masked, dlx, dly, dlz = voxelhelp.resamp_with_gradient(source_scan.voxels, rx[not_air_mask], ry[not_air_mask], rz[not_air_mask])
        resampled[not_air_mask] = resampled_masked
        matching_loss = np.mean((resampled - np.flip(resampled, axis=0))**2)
        if show_plots:
            voxelhelp.view_xyz_mips(resampled)
            print(f'matching loss: {matching_loss}')

        vf_resamp = vf3d.stack_scalar_fields(rx, ry, rz)
        L, vf_curve_loss_grad = vf3d.curvature_loss_with_gradient(vf_resamp)
        clx, cly, clz = vf3d.unstack_vector_to_xyz(vf_curve_loss_grad)

        difference = (resampled - np.flip(resampled, axis=0))[not_air_mask]
        mlx = np.zeros(headscan.voxels.shape, dtype=np.float32)
        mly = np.zeros(headscan.voxels.shape, dtype=np.float32)
        mlz = np.zeros(headscan.voxels.shape, dtype=np.float32)
        #print(mlx[not_air_mask].shape, difference.shape, dlx.shape)
        mlx[not_air_mask], mly[not_air_mask], mlz[not_air_mask] = difference*dlx*2/1000, difference*dly*2/1000, difference*dlz*2/1000
        #mlx, mly, mlz = match_error_grad(test_scan.voxels, rx, ry, rz)

        #k = voxelhelp.spherical_kernel(4)

        curvature_loss_multiplier = 2500
        matching_loss_multiplier = .0003

        blurr_grad_x = scipy.signal.convolve(-clx*curvature_loss_multiplier+ mlx*matching_loss_multiplier, k, mode='same') 
        blurr_grad_y = scipy.signal.convolve(-cly*curvature_loss_multiplier+ mly*matching_loss_multiplier, k, mode='same')
        blurr_grad_z = scipy.signal.convolve(-clz*curvature_loss_multiplier+ mlz*matching_loss_multiplier, k, mode='same')

        rx, ry, rz = rx - learn_rate*blurr_grad_x, ry - learn_rate*blurr_grad_y, rz - learn_rate*blurr_grad_z
        if show_plots:
            print(f'curvature loss: {L}')

        total_loss = curvature_loss_multiplier*L + matching_loss_multiplier*matching_loss

        if total_loss >= .99*last_loss:
            if show_plots:
                print(f'Gradient descent terminated at iteration {n+1}.')
        else:
            last_loss = total_loss


    resampled = voxelhelp.resamp_volume(source_scan.voxels, rx, ry, rz)
    if show_plots:
        if not fig_save_base_path==None:
            headscan.voxel_volumes = vf3d.cell_volume(vf3d.stack_scalar_fields(rx, ry, rz))/125
            voxelhelp.view_xyz_mips(np.clip(headscan.voxel_volumes*1000, .8, 1.25), 
                colorbars=True, save_path=os.path.join(fig_save_base_path, 'corrected_voxel_volumes.png'), 
                cmap='jet')
            voxelhelp.view_xyz_mips(resampled, 
                save_path=os.path.join(fig_save_base_path, 'corrected_headscan_mips.png'))
            voxelhelp.view_center_slices(voxelhelp.test_checker(rx, ry, rz), 
                CT_scale=False, save_path=os.path.join(fig_save_base_path, 'corrected_warp_checker.png'))
        else:
            voxelhelp.view_xyz_mips(resampled)
            voxelhelp.view_center_slices(voxelhelp.test_checker(rx, ry, rz), CT_scale=False)

    headscan.resamp_x = rx
    headscan.resamp_y = ry
    headscan.resamp_z = rz
    headscan.voxels = resampled

    return headscan


def segment_headscan(headscan, original_scan, teeth_seg, bone_seg, show_plots=False, fig_save_base_path=None):
    tissue_thresh = -200

    headscan.voxels[np.flip((headscan.resamp_y<0), axis=0)] = -1000


    # prepare compiled segmentation image
    split_scan_seg = deepcopy(bone_seg)
    for label in teeth_seg.labels[1:]:
        split_scan_seg.add_label(voxelhelp.mask_dilate(teeth_seg[label], 1), label)
    for union_label, union_sublabels in zip(teeth_seg.union_labels, teeth_seg.union_sublabels):
        split_scan_seg.new_union(union_label, union_sublabels)
    #

    # convert compiled segmentation to headspace
    compiled_seg = deepcopy(split_scan_seg)

    compiled_seg.int_image = voxelhelp.resamp_volume(split_scan_seg.int_image, headscan.resamp_x, headscan.resamp_y, headscan.resamp_z, mode='nearest', default_val=0)

    split_scan_seg.add_label((split_scan_seg.int_image==0)&(original_scan.voxels>-700), 'Tissue')

    bone_LR_map_error = np.sum((compiled_seg.int_image>0)^(np.flip(compiled_seg.int_image, axis=0)>0))/np.sum(compiled_seg.int_image>0)


    #arms = voxelhelp.mask_dilate(compiled_seg['Left Arm']|compiled_seg['Right Arm'], 1)
    arms = voxelhelp.mask_dilate((headscan.voxels > 350)&~compiled_seg['Skull'], 1)
    head = voxelhelp.mask_dilate(voxelhelp.mask_dilate(voxelhelp.largest_connected_only(voxelhelp.mask_erode(headscan.voxels>tissue_thresh, 6), connectivity=6), 4), 3)&(headscan.voxels>tissue_thresh)&~(arms)

    mean_voxel_volume = np.mean(headscan.voxel_volumes)
    tissue_volumes_left = np.sum((head&(headscan.x>0))*np.clip(headscan.voxel_volumes, .2*mean_voxel_volume, 5*mean_voxel_volume), axis=0)
    tissue_volumes_right = np.sum((head&(headscan.x<0))*np.clip(headscan.voxel_volumes, .2*mean_voxel_volume, 5*mean_voxel_volume), axis=0)
    neck_fade = np.clip(1-np.exp(-(headscan.y[0,:,:]-np.min(headscan.y[0,:,:]))*(np.cbrt(mean_voxel_volume)/5)), 0, 1)

    conf_kern_radius = 1/np.cbrt(mean_voxel_volume)
    conf_kern_z, conf_kern_y = np.meshgrid(np.arange(-conf_kern_radius, conf_kern_radius+1), np.arange(-conf_kern_radius, conf_kern_radius+1))
    conf_kern = np.clip(1-np.sqrt(conf_kern_z*conf_kern_z + conf_kern_y*conf_kern_y)/conf_kern_radius, 0, 1)
    conf_kern /= np.sum(conf_kern)

    tissue = head&(compiled_seg.int_image==0)

    edge_fade = 1-scipy.signal.convolve(scipy.signal.convolve(1*(tissue_volumes_right==0), conf_kern, mode='same'), conf_kern, mode='same')

    right_excess = (tissue_volumes_right-tissue_volumes_left)*neck_fade
    right_excess *= np.abs(right_excess) > np.cbrt(mean_voxel_volume)**2/2 # very small differences are likely due to imperfect left-right mapping, not tumor.

    tumor_volume = np.sum(right_excess)

    clip_fade_smooth_mask = scipy.signal.convolve(right_excess*scipy.signal.convolve(voxelhelp.largest_connected_only((right_excess*edge_fade)>np.cbrt(mean_voxel_volume)**2/.75), conf_kern, mode='same'), conf_kern, mode='same')
    clip_fade_smooth_mask *= voxelhelp.largest_connected_only(clip_fade_smooth_mask>0, connectivity=6)
    clip_fade_smooth_mask = np.minimum(clip_fade_smooth_mask, right_excess+2*np.cbrt(mean_voxel_volume)**2)

    tumor_volume = np.sum(clip_fade_smooth_mask)

    if np.sum(clip_fade_smooth_mask)>0:
        clip_fade_smooth_mask *= tumor_volume/np.sum(clip_fade_smooth_mask)

        tissue = head&(compiled_seg.int_image==0)

        tissue_volume_cumsum = np.cumsum(tissue*headscan.voxel_volumes, axis=0)

        tumor = tissue&(tissue_volume_cumsum-clip_fade_smooth_mask[np.newaxis,:,:]<=0)&tissue
        #tumor = voxelhelp.mask_dilate(voxelhelp.largest_connected_only(voxelhelp.mask_erode(tissue*(tissue_volume_cumsum-clip_fade_smooth_mask[np.newaxis,:,:]<=0), 3)), 1)&tissue
        #tumor_volume = np.sum(headscan.voxel_volumes*tumor)


        _, x_inv, y_inv, z_inv = voxelhelp.resamp_with_inverse(original_scan.voxels,
                                                               headscan.resamp_x,
                                                               headscan.resamp_y,
                                                               headscan.resamp_z)

        #split_space_tumor = voxelhelp.mask_dilate(voxelhelp.largest_connected_only(voxelhelp.mask_erode(voxelhelp.resamp_volume(tumor, x_inv, y_inv, z_inv, mode='nearest', default_val=0), 3)), 1)
        split_space_tumor = voxelhelp.resamp_volume(tumor, x_inv, y_inv, z_inv, mode='nearest', default_val=0)
        split_scan_seg.add_label(split_space_tumor, 'Tumor')

    else:
        tumor_volume = 0
        tumor = np.zeros(headscan.voxels.shape, dtype=bool)
        split_scan_seg.add_label(np.zeros(split_scan_seg.int_image.shape, dtype=bool), 'Tumor')

    if show_plots:
        if not fig_save_base_path==None:
            plt.figure(figsize=[3, 3], dpi=600)
        print('neck faded right - left')
        plt.imshow(np.flip((right_excess).T, axis=0), cmap='plasma')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.colorbar(label='Excess Tissue Volume (mm$^{3}$)')
        
        if not fig_save_base_path==None:
            plt.savefig(os.path.join(fig_save_base_path, 'right_excess_raw.png'), bbox_inches='tight')
            plt.figure(figsize=[3, 3], dpi=600)
        else:
            plt.show()
        print('neck fade, clip, smooth, mask')
        plt.imshow(np.flip(clip_fade_smooth_mask.T, axis=0), cmap='plasma')
        plt.colorbar(label='Excess Tissue Volume (mm$^{3}$)')
        plt.xlabel('y')
        plt.ylabel('z')
        if not fig_save_base_path==None:
            plt.savefig(os.path.join(fig_save_base_path, 'right_excess_smooth.png'), bbox_inches='tight')
        plt.show()

    compiled_seg.add_label(voxelhelp.mask_erode(tissue, 1), 'Normal Tissue')
    compiled_seg.add_label(tumor, 'Tumor')

    

    tumor_volume = np.sum(split_scan_seg['Tumor']*(.2**3))
    

    deformation_factor = np.mean(np.abs(np.log(np.clip(headscan.voxel_volumes/mean_voxel_volume, 1/5, 5))))

    #print(f'tumor volume: {tumor_volume}')
    #print(f'deformation factor: {deformation_factor}')

    warning_strings = []
    recommend_excluding = False
    if bone_LR_map_error > 1.2: # previously 1.1
        recommend_excluding = True
        warning_strings.append(f'It is recommended not to use this tumor volume because of high left-to-right bone mapping error of {bone_LR_map_error}')

    if (tumor_volume < -50) or (tumor_volume > 1300): # previously (-10, 1000)
        recommend_excluding = True
        warning_strings.append(f'It is recommended not to use this tumor volume because it had implausible value {tumor_volume}')

    if deformation_factor > .15: # previously .18
        recommend_excluding = True
        warning_strings.append(f'It is recommended not to use this tumor volume because for high deformation_factor {deformation_factor}. Headscan is likely twisted.')

    segmentation_info = {
        'tumor_volume' : tumor_volume,
        'deformation_factor' : deformation_factor,
        'bone_LR_map_error' : bone_LR_map_error,
        'warning_strings' : warning_strings,
        'recommend_excluding' : recommend_excluding
    }

    return compiled_seg, split_scan_seg, segmentation_info


def segment_head(scan, force_use_head_only=False):
    teeth_seg, teeth_points, tooth_seg_info = segment_teeth(scan)
    bone_seg, bone_points, bone_seg_info = segment_bones(scan, teeth_seg, teeth_points, use_head_only=force_use_head_only)
    headscan, resample_info = make_headscan(scan, teeth_seg, teeth_points, bone_seg, bone_points, use_head_only=bone_seg_info['use_head_only'])
    headscan = correct_headscan_by_gradient_descent(headscan, scan)
    headscan_seg, split_scan_seg, segmentation_info = segment_headscan(headscan, scan, teeth_seg, bone_seg)
    segmentation_info['tooth_seg_info'] = tooth_seg_info
    segmentation_info['bone_seg_info'] = bone_seg_info
    segmentation_info['resample_info'] = resample_info

    return headscan_seg, split_scan_seg, headscan, segmentation_info


def make_head_seg_figures(scan):
    teeth_seg, teeth_points, tooth_seg_info = segment_teeth(scan)
    teeth_fig_save_path = r'S:\RADONC\Karam_Lab\AutoContour\Paper\Figures\HeadSegmentationFigures\teeth_seg_mips.png'
    voxelhelp.view_xyz_mips(scan.voxels*(1+teeth_seg['Back Teeth']+2*teeth_seg['Bottom Teeth']+2*teeth_seg['Front Teeth']),
                            annotation=teeth_points, label_points=True, save_path=teeth_fig_save_path)

    bone_seg, bone_points, bone_seg_info = segment_bones(scan, teeth_seg, teeth_points)
    bone_fig_save_path = r'S:\RADONC\Karam_Lab\AutoContour\Paper\Figures\HeadSegmentationFigures\bone_seg_mips.png'
    voxelhelp.view_xyz_mips(scan.voxels*(2+bone_seg['Left Arm']+2*bone_seg['Right Arm']+2*bone_seg['Cranium']),
                            annotation=bone_points, label_points=True, save_path=bone_fig_save_path)

    headscan, resample_info = make_headscan(scan, teeth_seg, teeth_points, bone_seg, bone_points, 
        show_plots=True, fig_save_base_path=r'S:\RADONC\Karam_Lab\AutoContour\Paper\Figures\HeadSegmentationFigures')
    headscan = correct_headscan_by_gradient_descent(headscan, scan,
        show_plots=True, fig_save_base_path=r'S:\RADONC\Karam_Lab\AutoContour\Paper\Figures\HeadSegmentationFigures')

    headscan_seg, split_scan_seg, segmentation_info = segment_headscan(headscan, scan, teeth_seg, bone_seg,
                        fig_save_base_path=r'S:\RADONC\Karam_Lab\AutoContour\Paper\Figures\HeadSegmentationFigures',
                        show_plots=True)

if __name__ == "__main__":
    test_of_segment_head()
