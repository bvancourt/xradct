import numpy as np
import importlib
import warnings
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import linalg

import voxelhelp
importlib.reload(voxelhelp)


class CTScan(): # should be compatible with both XRad and Quantum GXII uCT
    def __init__(self, voxels, voxel_spacing, name):
        self.name = name
        self.spacing = voxel_spacing # in millimeters
        self.voxels = voxels


class XRadCT(CTScan):
    def __init__(self, voxels, name):
        super().__init__(voxels, [.1, .1, .1], name)
        self.x, self.y, self.z = voxelhelp.index_grid(voxels)

    @classmethod
    def from_dicom_files(cls, path_to_files):
        voxels = voxelhelp.voxels_from_dicom_folder(path_to_files)
        name = path_to_files.split('\\')[-1].split('/')[-1]
        return cls(voxels, name)

    def make_bed_air(self):
        test_collumn_spacing = 8
        extra_deletion_margin = 4

        bed_points_x = []
        bed_points_y = []
        bed_points_z = []

        for i in range(0, self.voxels.shape[0], test_collumn_spacing):
            for j in range(0, self.voxels.shape[1], test_collumn_spacing):
                col_mean = np.mean(self.voxels[i,j,:])
                sig.convolve(self.voxels[i,j,:], np.ones(1),mode = 'same')
                bed_height = np.argmin(np.diff(self.voxels[i, j, :]))
                if col_mean <= -900:
                    bed_points_x.append(i)
                    bed_points_y.append(j)
                    bed_points_z.append(bed_height)

        bed_points = np.array(list(zip(bed_points_x, bed_points_y, bed_points_z)))

        # matrix for least-squares plane fitting
        A = np.transpose(np.stack([bed_points[:,0], bed_points[:,1], np.ones(bed_points.shape[0])]))

        # z_fit = (a, b, c), where a*x + b*y + c ~= z
        z_fit, *_ = linalg.lstsq(A, np.transpose(bed_points[:,2]), rcond=None)

        # normal vector to plane of bed
        nhat = np.array([-z_fit[0], -z_fit[1], 1])/linalg.norm(np.array([-z_fit[0], -z_fit[1], 1]))
        nhat *= -np.sign(np.dot(nhat, np.ones(3)))

        center_point = np.mean(bed_points, axis=0)

        below_plane = ((self.x - center_point[0])*nhat[0] + (self.y - center_point[1])*nhat[1] + (self.z - center_point[2])*nhat[2]) > - extra_deletion_margin

        self.voxels_with_bed = copy.deepcopy(self.voxels) # Saves an unedited copy of the voxel array for safety. This could be commented to save RAM
        self.voxels[below_plane] = -1000


class OneMouseScan(XRadCT):
    def remove_xtra_mouse(self): #This function removes the extra mouse pieces from a scan which has been cropped, but contains parts of other mice in it.
        mouse_labels = cc3d.connected_components(self.voxels > -700)
        stats = cc3d.statistics(mouse_labels)
        voxel_counts = stats['voxel_counts']
        sorted_indices = np.argsort(voxel_counts)[::-1]

        self.raw_voxels = copy.deepcopy(self.voxels)# Saves an unedited copy of the voxel array for safety. This could be commented to save RAM

        for i in sorted_indices[2:]:
            self.voxels[mouse_labels == i] = -1000

        return self


class OneRowScan(XRadCT):
    def __init__(self, voxels, name):
        super().__init__(voxels, name)
        self.mouseIDs = [with_spaces.replace(' ', '') for with_spaces in name.split(',')]

    def to_individual_mice(self):
        def cosine_window(length):
            t = np.arange(1, length)
            return (1-np.cos(2*np.pi*t/length))/length

        mouseness = np.mean(self.voxels, axis = (0,2))# 1D signal along the direction to split mice

        pad_number = 5000 # effectively a default CT number for locations outside the scan
        smooth_mouse = sig.convolve(mouseness-pad_number, cosine_window(250),mode = 'same')+pad_number

        min_list = []

        for i in range(1, smooth_mouse.shape[0]-1): 
            if smooth_mouse[i-1]> smooth_mouse[i] and smooth_mouse[i+1]> smooth_mouse[i]:
                min_list.append((i, smooth_mouse[i]))

        #The second part of this function actually splits the scans by defining the y-values of the minima of the curve to be the y values to crop at.
        #These functions are defined for 1, 2, and 3 mouse scans currently.
        split_points = [min_tuple[0] for min_tuple in min_list[1:-1]]

        voxel_arrays = []
        if len(split_points)+1 != len(self.mouseIDs):
            warnings.warn(f'expected {len(self.mouseIDs)} mice but detected {len(split_points)+1}')
            plt.plot(smooth_mouse)
            print(f'split points: {split_points}')

        #One-mouse: returns the whole scan
        elif len(split_points) == 0:
            voxel_arrays.append(remove_xtra_mouse(self.voxels))

        #Two-mouse: returns the scan split in 2 at the minima.
        elif len(split_points) == 1:
            left_mouse = self.voxels[:,:split_points[0], :]
            right_mouse = self.voxels[:,split_points[0]:, :]
            
            voxel_arrays.append(remove_xtra_mouse(left_mouse))
            voxel_arrays.append(remove_xtra_mouse(right_mouse))

        #3-mouse, returns the three separate volumes. 
        elif len(split_points) == 2:
            left_mouse = self.voxels[:,:split_points[0], :]
            mid_mouse = self.voxels[:,split_points[0]:split_points[1], :]
            right_mouse = self.voxels[:,split_points[1]:, :]

            voxel_arrays.append(remove_xtra_mouse(left_mouse))
            voxel_arrays.append(remove_xtra_mouse(mid_mouse))
            voxel_arrays.append(remove_xtra_mouse(right_mouse))

        else:
            warnings.warn('This function only supports scans with 1, 2, or 3 mice. No split scans produced.')
            
            one_mouse_scans = [OneMouseScan(voxels, name).remove_xtra_mouse() for voxels, name in zip(voxel_arrays, self.mouseIDs)]

        return one_mouse_scans
