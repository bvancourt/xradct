import numpy as np
import nibabel as nib
import importlib
import warnings
import scipy.signal
import matplotlib.pyplot as plt
from numpy import linalg
import copy
import os
import cc3d

import voxelhelp
importlib.reload(voxelhelp)


class CTScan(): # should be compatible with both XRad and Quantum GXII uCT
    def __init__(self, voxels, voxel_spacing, name, date):
        self.name = name
        self.voxel_spacing = voxel_spacing # in millimeters
        self.voxels = voxels
        self.date = date

    def gaussian_LP(self, sigma_mm, extent_sigma=3, pad_value=-1000):
        kernel = voxelhelp.gaussian_filter_mm(sigma_mm, self.voxel_spacing, extent_mm)
        kernel /= np.sum(kernel)
        return scipy.signal.convolve(self.voxels-pad_value, kernel, mode='same')+pad_value

    def gaussian_HP(self, sigma_mm, extent_sigma=3, pad_value=-1000):
        return gaussian_LP(self, sigma_mm, extent_sigma=extent_sigma, pad_value=pad_value) - self.voxels

    def nudge_CT_numbers(self):
        self.voxels = voxelhelp.nudge_CT_numbers(self.voxels)


class XRadCT(CTScan):
    def __init__(self, voxels, voxel_spacing, name, date):
        super().__init__(voxels, voxel_spacing, name, date)
        self.x, self.y, self.z = voxelhelp.index_grid(voxels)

    @classmethod
    def from_dicom_files(cls, path_to_files):
        file_list = os.listdir(path_to_files)
        mouse_IDs = file_list[0].split('_')[0]
        voxels, voxel_spacing, date = voxelhelp.voxels_from_dicom_folder(path_to_files)
        name = [path_to_files.split('\\')[-1].split('/')[-1], mouse_IDs]

        return cls(voxels, voxel_spacing, name, date)

    @classmethod
    def from_xradct(cls, xradct_object):
        return cls(xradct_object.voxels, xradct_object.voxel_spacing, xradct_object.name, xradct_object.date)

    @classmethod
    def from_nii(cls, path_to_file):
        nii_object = nib.load(path_to_file)
        voxel_spacing = (nii_object.header['pixdim'][3],
                         nii_object.header['pixdim'][1],
                         nii_object.header['pixdim'][2])
        voxels = nii_object.get_fdata().transpose([2,0,1])
        return cls(voxels, voxel_spacing, [os.path.split(path_to_file)[-1].split('.')[0], os.path.split(path_to_file)[-1].split('.')[0]], 'date not read form nii header')

    def halve_resolution(self):
        # Fixed this function 11-27-23
        self.big_voxel_array = copy.copy(self.voxels)
        self.y, self.x, self.z =  np.meshgrid(np.arange(self.big_voxel_array.shape[1]//2),
                                  np.arange(self.big_voxel_array.shape[0]//2),
                                  np.arange(self.big_voxel_array.shape[2]//2))

        small_voxel_array = np.zeros(self.x.shape)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    small_voxel_array += self.big_voxel_array[self.x*2+i, self.y*2+j, self.z*2+k]/8

        self.voxels = np.round(small_voxel_array).astype(self.big_voxel_array.dtype)

        self.voxel_spacing = tuple(np.array(self.voxel_spacing)*2)

        return self

    def unhalve_resolution(self):
        self.downsampled_voxels = copy.copy(self.voxels)
        self.voxels = copy.copy(self.big_voxel_array)
        delattr(self, 'big_voxel_array')

        self.x, self.y, self.z = voxelhelp.index_grid(self.voxels)
        self.voxel_spacing = tuple(np.array(self.voxel_spacing)/2)

        return self

    def remove_bed(self):
        pointxlist = []
        pointylist = []
        pointzlist = []
        irange = self.voxels.shape[0]
        jrange = self.voxels.shape[1]
        for i in range(0, irange, 10):
            for j in range(0, jrange,10):
                mean = np.mean(self.voxels[i,j,:])
                scipy.signal.convolve(self.voxels[i,j,:], np.ones(1),mode = 'same')
                d = np.argmin(np.diff(self.voxels[i, j, :]))
                if mean <= -850:
                    pointxlist.append(i)
                    pointylist.append(j)
                    pointzlist.append(d)
                else:
                    mean = mean

        #print('point lists done')
        X = np.array(pointxlist)
        Y = np.array(pointylist)
        Z = np.array(pointzlist)
        pointsList = list(zip(pointxlist, pointylist, pointzlist))
        Points = np.array(pointsList)

        center, normal = voxelhelp.plane_near_points(Points)

        below_plane = ((self.x - center[0])*normal[0]+(self.y-center[1])*normal[1]+(self.z - center[2])*normal[2])*np.sign(normal[2]) > 0

        self.voxels_with_bed = copy.deepcopy(self.voxels) # Saves an unedited copy of the voxel array for safety. This could be commented to save RAM
        self.voxels[~below_plane] = -1000


    def remove_iso_tube(self):
        tube_thresh = 500

        mask = (self.voxels>tube_thresh)
        cc3d_labels = cc3d.connected_components(mask, connectivity=6)
        cc3d_stats = cc3d.statistics(cc3d_labels)
        voxel_counts = cc3d_stats['voxel_counts']
        sorted_indices = np.argsort(voxel_counts)[::-1]
        for index in sorted_indices:
            candidate_tube = (cc3d_labels==index)
            if np.any(candidate_tube&mask)&(np.max(self.y[candidate_tube]>=self.voxels.shape[1]-4)&(np.min(self.y[candidate_tube]>=self.voxels.shape[1]-70))):
                self.voxels[voxelhelp.mask_dilate(candidate_tube, 2)] = -1000
                return self
        return self


    def remove_xtra_mouse(self): #This function removes the extra mouse pieces from a scan which has been cropped, but contains parts of other mice in it.
        mouse_labels = cc3d.connected_components(self.voxels > -600)#default 600
        #print('mouse_labels')
        stats = cc3d.statistics(mouse_labels)
        voxel_counts = stats['voxel_counts']
        sorted_indices = np.argsort(voxel_counts)[::-1]
        #print('sorted indices')
        #print(sorted_indices.shape)

        #self.raw_voxels = copy.deepcopy(self.voxels)# Saves an unedited copy of the voxel array for safety. This could be commented to save RAM

        for i in sorted_indices[2:100]:
            self.voxels[mouse_labels == i] = -1000

        return self


class Segmentation:
    def __init__(self, int_image, labels, union_labels=[], union_sublabels=[]):
        self.int_image = int_image.astype(np.uint16)
        self.labels = labels
        self.union_labels = union_labels
        self.union_sublabels = union_sublabels

    @classmethod
    def empty(cls, shape):
        return cls(np.zeros(shape, dtype=np.uint16), ['Unlabeled'], union_labels=[], union_sublabels=[])

    @property
    def shape(self):
        return self.int_image.shape

    def add_label(self, mask, label):
        if (label in self.labels):
            print(f"label '{label}' could not be added to segmentation, because it was already in the label list.")
        else:
            self.int_image[mask] = len(self.labels)
            self.labels.append(label)

    def add_label_over_existing(self, old_label, mask, new_label):
        if not old_label in self.labels:
            print(f"Label '{old_label}' not in segmentation")
        else:
            if new_label in self.labels:
                print(f"label '{label}' could not be added to segmentation, because it was already in the label list.")
            else:
                self.int_image[mask&self[old_label]] = len(self.labels)
                self.labels.append(new_label)

    def subdivide_label(self, target_label, sublabels, mask):
        if target_label in self.union_labels:
            print(f'Target label {target_label} was already a union, and therefore could not be subdivided.')
        elif any([label in self.labels for label in sublabels]):
            print(f'Target label {target_label} was not subdivided, because at least one of the sublabels {sublabels} was already defined.')
        else:
            self.union_labels.append(target_label)
            self.union_sublabels.append(sublabels)
            self.labels[self.labels.index(target_label)] = sublabels[0]
            self.add_label_over_existing(sublabels[0], mask, sublabels[1])


    def new_union(self, union_label, sublabels):
        if union_label in self.union_labels:
            print(f'Union {union_label} could not be defined, because a union already existed with that name.')
        else:
            self.union_labels.append(union_label)
            self.union_sublabels.append(sublabels)
            for label in sublabels:
                if not label in self.labels:
                    print(f'Warning! Union {union_label} was defined to include undefined label {label}.')

    def __getitem__(self, label):
        if not label in self.labels+self.union_labels:
            print(f"Warning! Label '{label}' not in segmentation.")
            return np.zeros(self.shape, dtype=bool)
        elif label in self.labels:
            return self.int_image==self.labels.index(label)
        elif label in self.union_labels:
            union_index = self.union_labels.index(label)
            ouput_mask = np.zeros(self.shape, dtype=bool)
            for sublabel in self.union_sublabels[union_index]:
                ouput_mask |= self[sublabel]
            return ouput_mask


class PointAnnotationSet:
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels

    @classmethod
    def empty(cls):
        return cls([],[])

    def add_point(self, point, label):
        self.points.append(point)
        self.labels.append(label)

    def __add__(self, other): # concatenates two point sets to produce a new object
        return type(self)(self.points+other.points, self.labels+other.labels)

    def __getitem__(self, label):
        if not label in self.labels:
            print(f"Label '{label}' not in point set.")
            return (np.nan,np.nan,np.nan)
        else:
            if self.labels.count(label) > 1:
                temp_labels = copy.deepcopy(self.labels)
                temp_points = copy.deepcopy(self.points)
                return temp_points[temp_labels.index(label)]
                print(f'Point set contained {self.labels.count(label)} instances of label "{label}." Only the last point added is returned.')
            else:
                return self.points[self.labels.index(label)]
