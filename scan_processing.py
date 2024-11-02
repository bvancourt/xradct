#VERSION NOTES (version 3)

#OVERVIEW:
#this file is a python module containing functions that allow .dicom files of three- and five-mouse scans to be processed in a way that largely isolates the individual mice.
#the XRadCT class is imported from the module "xradct.py" which must be in the same folder location
#The process is as follows: Scans are uploaded into an array using pandas. Whether it is a five or three mouse scan is determined, and the proper class is assigned.
# Once a class is assigned the class methods can be applied and the scans will be split, and the bed will be removed.
# After the one mouse scan objects are returned, the clean up is done, and the extra mouse in the image, such as legs and ears, is removed.
#one mouse scans with the mouse ID attached are returned.

#CHANGES FROM VERSION 1:
#The biggest change was moving the bed removal from the OneMouseScan class to the individual three- and 5- mouse classes. For some reason it works more quickly this way.
#most extra comments have been removed.
# in the file folder this is stored in, "untitled" jupyter file has use guide and "test data" includes some test scans of both three- and five-mouse scans.

#CHANGES FROM VERSION 2:
#Added side removal function
# The way I am hoping this works is by finding the points at the front of each side piece, and subtracting a plane from the image
# Additionally, added features to process files with both 0.1mm and 0.2mm voxels.
# A function for removing the front parts of the mouse holder was also added, with partial success.
# One main bug is that for the 5-mouse scans with 0.1mm spacing, for some reason it splits the scans incorrectly, and then removes the sides incorrectly. I believe it removes the whole scan.






import numpy as np
import importlib
import warnings
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import linalg
import copy
import os
import cc3d
import logging
##############################################

import xradct
import voxelhelp
from xradct import XRadCT

class OneMouseScan(XRadCT):

    def test_empty(self, show_plots = False): #tests if a one-mouse scan contains a mouse
        if np.sum(self.voxels>-200) > 290317:
            return True
        else:
            return False

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

    def remove_sides(self): #meant to try to remove little side pieces in 5-mouse scans but:/ # NEED to fix numbers at some point
        #Step 1: find sides by taking a sample of top 20 voxels in the z direction

        max = []
        for i in range(self.voxels.shape[2]-45, self.voxels.shape[2]-25):
            max_value = np.max(self.voxels[:,self.voxels.shape[1]-30, i])
            max.append(max_value)
            base_y = int(self.voxels.shape[1]-30)
            base_z = int(self.voxels.shape[2]-35)


        #print(max)
        if np.mean(np.array(max)) > -300:
            sides_found = True

            side_finder_row = self.voxels[:, base_y,  base_z]

            relative_maxima = np.array(sig.argrelextrema(side_finder_row, np.greater))

            #print(relative_maxima)
            top_side_location_x = []
            for i in relative_maxima[0]:

                if side_finder_row[i] > -300:
                    top_side_location_x.append(i)
                else:
                    pass

            if len(top_side_location_x) > 1:
                top_x_coords = np.array(top_side_location_x)
            else:
                warnings.warn('One side detected. Opposite will be approximated')
                if top_side_location_x[0] > self.voxels.shape[0]//2:
                    left_x = relative_maxima[0,3]
                    top_x_coords = np.array([left_x, top_side_location_x[0]])
                else:
                    right_x = relative_maxima[0,-3]
                    top_x_coords = np.array([top_side_location_x[0], right_x])


            left_plane_coords = np.array([[top_x_coords[0]+5, base_y, base_z],[top_x_coords[0]+17, base_y, base_z-30], [top_x_coords[0]+5, base_y-int(self.voxels.shape[1]/2), base_z],[top_x_coords[0]+15, base_y-int(self.voxels.shape[1]/2), base_z-30]])
            right_plane_coords = np.array([[top_x_coords[1]-5, base_y, base_z],[top_x_coords[1]-17, base_y, base_z-30], [top_x_coords[1]-5, base_y-int(self.voxels.shape[1]/2), base_z],[top_x_coords[1]-15, base_y-int(self.voxels.shape[1]/2), base_z-30]])


            #left plane calculation
            x, y = np.meshgrid(range(self.voxels.shape[0]-1), range(self.voxels.shape[0]-1))

            xx = x.flatten()
            yy = y.flatten()

            #print(x.shape, xx.shape)

            A = np.c_[left_plane_coords[:,0], left_plane_coords[:,1], np.ones(left_plane_coords.shape[0])]
            C,_,_,_ = scipy.linalg.lstsq(A, left_plane_coords[:,2])

            zz = C[0]*x + C[1]*y + C[2]



            points = np.array(list(zip(xx, yy, zz.flatten())))



            for i in points:
                x = int(i[0])
                y = int(i[1])
                z = int(i[2])
                if 0 < z < self.voxels.shape[2] and y < self.voxels.shape[1]:
                    self.voxels[:x, :, :z] = -1000






            # right plane calculation
            xr, yr = np.meshgrid(range(self.voxels.shape[0]-1), range(self.voxels.shape[0]-1))

            xxr = xr.flatten()
            yyr = yr.flatten()

            #print(xr.shape, xxr.shape)

            Ar = np.c_[right_plane_coords[:,0], right_plane_coords[:,1], np.ones(right_plane_coords.shape[0])]
            Cr,_,_,_ = scipy.linalg.lstsq(Ar, right_plane_coords[:,2])

            zzr = Cr[0]*xr + Cr[1]*yr + Cr[2]



            r_points = np.array(list(zip(xxr, yyr, zzr.flatten())))



            for i in r_points:
                xr = int(i[0])
                yr = int(i[1])
                zr = int(i[2])
                if 0 < zr < self.voxels.shape[2]-1 and yr < self.voxels.shape[1]:
                    self.voxels[xr:, :, :zr] = -1000

        else:
            sides_found = False

        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111)
        # ax2.imshow(np.max(self.voxels, axis = 0))
        # plt.show()
        return self

    def remove_fronts(self):


        sides_found = False
        max = []

        for i in range(self.voxels.shape[2]-35, self.voxels.shape[2]-20):
            if self.voxels.shape[1]-80 > 0:
                max_value = np.max(self.voxels[self.voxels.shape[0]-20,self.voxels.shape[1]-80:, i])
            else:
                max_value = np.max(self.voxels[self.voxels.shape[0]-20,:, i])
            max.append(max_value)
            base_x = int(self.voxels.shape[0]//2)
            base_z = int(self.voxels.shape[2]-20)

        if np.mean(np.array(max)) > -300:
            front_found = True

            #window = self.voxels.shape[0]/20
            #pad_number = 5000 # effectively a default CT number for locations outside the scan
            #t = np.arange(1, window)
            #cosine_window = (1-np.cos(2*np.pi*t/window))/window

            front_finder_row = self.voxels[base_x,:, base_z]
            #smooth_side_finder = sig.convolve(side_finder_row-pad_number, cosine_window, mode = 'same')+pad_number


            relative_maxima = sig.argrelextrema(front_finder_row, np.greater)
            rm = np.array(relative_maxima)
            front_location_y = []

            for i in rm[0]:

                if front_finder_row[i] > -500:
                    front_location_y.append(i)
                else:
                    pass
            fly = np.array(front_location_y)
            if fly.shape[0] > 0:
                front = np.max(fly)
                if front > 75:
                    self.voxels[:, front:, :] = -1000
            else:
                pass

        else:
            pass

    # def get_max_intensity(self, path_to_save_to, file_name):
    #     image = self.voxels
    #     matplotlib.image.imsave(str(path_to_save_to) + str(file_name)+ '.jpg', image, cmap = 'bone')
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

    def clean_up_scan(self, skip_sides = False, skip_fronts = False):
        self.voxels = voxelhelp.nudge_CT_numbers(self.voxels)
        mouse = self.test_empty()
        if mouse == True:
            self.remove_xtra_mouse()
            if skip_sides == False:
                self.remove_sides()
            if skip_fronts == False:
                self.remove_fronts()
            self.remove_iso_tube()
            self.empty = False

            return self

        elif mouse == False:
            print("WARNING: Scan may not contain a mouse.")
            #return self.name
            try:
                self.remove_xtra_mouse()
                if skip_sides == False:
                    self.remove_sides()
                if skip_fronts == False:
                    self.remove_fronts()
                self.remove_iso_tube()
            except BaseException:
                logging.exception("ERROR: Possible empty scan - not cleaned up.")

            self.empty = True

            return self


#OneMouseScan(voxels, self.voxel_spacing, name, self.date)
class OneRowScan(XRadCT):
    def __init__(self, voxels, name, date):
        super().__init__(voxels, name, date)

    def to_individual_mice(self, show_plots=False):
        if self.voxel_spacing[0] == 0.1:
            self.halve_resolution()
            self.x, self.y, self.z = voxelhelp.index_grid(self.voxels)
        else:
            pass

        # plt.imshow(self.voxels[:, 150, :], cmap = "bone")
        # plt.show()

        def cosine_window(length):
            t = np.arange(1, length)
            return (1-np.cos(2*np.pi*t/length))/length

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
        #
        # bed_heights=[]
        # for i in Z:
        #     if i < 50:
        #         bed_heights.append(i)
        #
        # bed = int(np.mean(np.array(bed_heights)))
        print(self.x.shape, self.voxels.shape[0])

        self.voxels_with_bed = copy.deepcopy(self.voxels) # Saves an unedited copy of the voxel array for safety. This could be commented to save RAM
        self.voxels[~below_plane] = -1000

        mouseIDs = [self.name[0] + with_spaces.replace(' ', '') for with_spaces in self.name[1].split(',')]

        mouseness = np.mean(self.voxels, axis = (1,2))# 1D signal along the direction to split mice

        pad_number = 5000 # effectively a default CT number for locations outside the scan
        smooth_mouse = sig.convolve(mouseness-pad_number, cosine_window(125), mode = 'same')+pad_number

        # plt.plot(smooth_mouse)
        # plt.title("Smoothed Density of yz Slices Over x Axis")
        # plt.ylabel("Mean Density")
        # plt.xlabel("X")
        # plt.show()

        min_list = []

        for i in range(1, smooth_mouse.shape[0]-1):
            if smooth_mouse[i-1]> smooth_mouse[i] and smooth_mouse[i+1]> smooth_mouse[i]:
                min_list.append((i, smooth_mouse[i]))

        if show_plots:
            plt.plot(smooth_mouse)
            for min_point in min_list:
                plt.plot([min_point[0],min_point[0]], [-1000, 1000])
            plt.show()

        #The second part of this function actually splits the scans by defining the y-values of the minima of the curve to be the y values to crop at.
        #These functions are defined for 1, 2, and 3 mouse scans currently.
        split_points = [min_tuple[0] for min_tuple in min_list[1:-1]]

        toothness = np.mean(self.voxels > 2000, axis = (1, 2))
        smooth_tooth = sig.convolve(toothness, cosine_window(100), mode = 'same')
        max_list = []
        for i in range(1, smooth_tooth.shape[0]-1):
            if smooth_tooth[i-1] < smooth_tooth[i] and smooth_tooth[i+1] < smooth_tooth[i]:
                max_list.append((i, smooth_tooth[i]))
        tooth_points = [max_tuple[0] for max_tuple in max_list]

        # plt.plot(smooth_tooth)
        # plt.title("Smoothed Density Maximum of yz Slices Over x Axis")
        # plt.ylabel("Max. Density of yz Slice")
        # plt.xlabel("X")
        # plt.show()



        voxel_arrays = []
        if len(split_points)+1 != len(mouseIDs):
            warnings.warn(f'expected {len(mouseIDs)} mice but detected {len(split_points)+1}')
            plt.plot(smooth_mouse)
            print(f'split points: {split_points}')
            plt.ion()

            #splits scans a different way.
            print("Trying splitting method 2...")
            toothness = np.mean(self.voxels > 2000, axis = (1, 2))

            smooth_tooth = sig.convolve(toothness, cosine_window(100), mode = 'same')

            max_list = []
            for i in range(1, smooth_tooth.shape[0]-1):
                if smooth_tooth[i-1] < smooth_tooth[i] and smooth_tooth[i+1] < smooth_tooth[i]:
                    max_list.append((i, smooth_tooth[i]))

            tooth_points = [max_tuple[0] for max_tuple in max_list]

            if len(tooth_points) == 0:
                warnings.warn("No mice in scan")
                plt.plot(smooth_tooth)
                plt.ion()

            elif len(tooth_points) == 1:
                print("One mouse found")
                if tooth_points[0]-90 > 0 and tooth_points[0]+90 < self.voxels.shape[0]:
                    voxel_arrays.append(self.voxels[tooth_points[0]-90:tooth_points[0]+90, :, :])
                else:
                    voxel_arrays.append(self.voxels[:,:,:])

            elif len(tooth_points) == 2:
                #print("Two mice found")
                if tooth_points[0] - 90 >0 and tooth_points[0]+90 < self.voxels.shape[0]:
                    left_mouse = self.voxels[tooth_points[0] - 90:tooth_points[0] + 90, :, :]
                    right_mouse = self.voxels[tooth_points[1] - 90:tooth_points[1] + 90, :, :]
                    voxel_arrays.append(left_mouse)
                    voxel_arrays.append(right_mouse)
                else:
                    left_mouse = self.voxels[:tooth_points[0] + 90, :, :]
                    right_mouse = self.voxels[tooth_points[1] - 90:, :, :]
                    voxel_arrays.append(left_mouse)
                    voxel_arrays.append(right_mouse)

                #print("Mice separated successfully")

            elif len(tooth_points) == 3:
                #print("Three mice found")

                left_mouse = self.voxels[:tooth_points[0] + 90, :, :]
                center_mouse = self.voxels[tooth_points[1] - 90:tooth_points[1] + 90, :, :]
                right_mouse = self.voxels[tooth_points[2] - 90:, :, :]
                voxel_arrays.append(left_mouse)
                voxel_arrays.append(center_mouse)
                voxel_arrays.append(right_mouse)


        #One-mouse: returns the whole scan
        elif len(split_points) == 0:
            if tooth_points[0]-90 > 0 and tooth_points[0]+90 < self.voxels.shape[0]:
                voxel_arrays.append(self.voxels[tooth_points[0]-90:tooth_points[0]+90, :, :])
            else:
                voxel_arrays.append(self.voxels[:,:,:])
            print("Scans split, one mouse found")

        #Two-mouse: returns the scan split in 2 at the minima.
        elif len(split_points) == 1:

            if len(tooth_points) == 2:
                half_mouse_1_size = split_points[0] - tooth_points[0]
                half_mouse_2_size = tooth_points[1] - split_points[0]
                mouse_1_size = 2*half_mouse_1_size+5
                mouse_2_size = 2*half_mouse_2_size+5
                if split_points[0]-mouse_1_size > 0 and (split_points[0] + mouse_2_size < self.voxels.shape[0]):
                    left_mouse = self.voxels[split_points[0]-mouse_1_size:split_points[0], :, :]
                    right_mouse = self.voxels[split_points[0]:split_points[0]+mouse_2_size,:, :]
                    voxel_arrays.append(left_mouse)
                    voxel_arrays.append(right_mouse)
                    print("scans split",2, "mice found")
                else:
                    left_mouse = self.voxels[:split_points[0], :, :]
                    right_mouse = self.voxels[split_points[0]:,:, :]
                    voxel_arrays.append(left_mouse)
                    voxel_arrays.append(right_mouse)
                    print("scans split",2, "mice found")
            else:
                print("Unreliable tooth points")

                left_mouse = self.voxels[:split_points[0], :, :]
                right_mouse = self.voxels[split_points[0]:,:, :]
                voxel_arrays.append(left_mouse)
                voxel_arrays.append(right_mouse)
            print("scans split",2, "mice found")

        #3-mouse, returns the three separate volumes.
        elif len(split_points) == 2:

            if len(tooth_points) == 3:
                half_mouse_1_size = split_points[0] - tooth_points[0]
                half_mouse_2_size = tooth_points[1] - split_points[0]
                half_mouse_3_size = tooth_points[2] - split_points[1]
                mouse_1_size = 2*half_mouse_1_size
                print(mouse_1_size)
                mouse_2_size = 2*half_mouse_2_size
                mouse_3_size = 2*half_mouse_3_size

                if split_points[0]-mouse_1_size > 0 and split_points[1] + mouse_3_size < self.voxels.shape[0]:
                    left_mouse = self.voxels[split_points[0]-mouse_1_size:split_points[0],:, :]
                    mid_mouse = self.voxels[split_points[0]:split_points[1], :, :]
                    right_mouse = self.voxels[split_points[1]:split_points[1]+mouse_3_size, :, :]
                    voxel_arrays.append(left_mouse)
                    voxel_arrays.append(mid_mouse)
                    voxel_arrays.append(right_mouse)
                    print("scans split",3, "mice found")
                else:
                    left_mouse = self.voxels[:split_points[0],:, :]
                    mid_mouse = self.voxels[split_points[0]:split_points[1], :, :]
                    right_mouse = self.voxels[split_points[1]:, :, :]
                    voxel_arrays.append(left_mouse)
                    voxel_arrays.append(mid_mouse)
                    voxel_arrays.append(right_mouse)
                    print("scans split",3, "mice found")

            else:
                print("Unreliable tooth points")

                left_mouse = self.voxels[:split_points[0],:, :]
                mid_mouse = self.voxels[split_points[0]:split_points[1], :, :]
                right_mouse = self.voxels[split_points[1]:, :, :]
                voxel_arrays.append(left_mouse)
                voxel_arrays.append(mid_mouse)
                voxel_arrays.append(right_mouse)
                print("scans split",3, "mice found")

        else:
            warnings.warn('This function only supports scans with 1, 2, or 3 mice. No split scans produced.')

        for array in voxel_arrays:
            # plt.imshow(array[:, 150, :], cmap = "bone")
            # plt.show()
            if array.shape[0]<50:
                plt.imshow(array[:,5,:], cmap = "bone")
                warnings.warn("This scan was split improperly.")

        one_mouse_scans = [OneMouseScan(voxels, self.voxel_spacing, name, self.date) for voxels, name in zip(voxel_arrays, mouseIDs)]

        return one_mouse_scans

    def get_individual_mice(self):

        self.remove_bed()
        # plt.imshow(self.voxels[:, 150, :], cmap = "bone")
        self.to_individual_mice()



class FiveMouseScan(XRadCT):
    def __init__(self, voxels, voxel_spacing, name, date):
        super().__init__(voxels, voxel_spacing, name, date)


    def to_single_mice(self):

         # this function will find the planes at which the various beds in the image are, and these will be used to split the scan.
        if self.voxel_spacing[0] == 0.1:
            self.halve_resolution()
        else:
            pass

        # plt.imshow(self.voxels[:, 150, :], cmap = "bone")
        # plt.show()

        def cosine_window(length):
            t = np.arange(1, length)
            return (1-np.cos(2*np.pi*t/length))/length

        #print(self.name[0])

        bed_find_1 = 100
        bed_find_2 = 200
        mouse_wide = 100


        mouseIDs = [self.name[0] + with_spaces.replace(' ', '') for with_spaces in self.name[1].split(',')]
        print(mouseIDs)
        date = self.date


        pointxlist = []
        pointylist = []
        pointzlist = []
        irange = self.voxels.shape[0]
        jrange = self.voxels.shape[1]
        for i in range(0, irange, 5):
            for j in range(0, jrange,5):
                mean = np.mean(self.voxels[i,j,:])
                #scipy.signal.convolve(self.voxels[i,j,:], np.ones(1),mode = 'same')
                d = np.argmin(np.diff(self.voxels[i, j, :]))
                if mean <= -900:
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

        bed_heights=[]
        platform_1_mouse = []
        platform_2_mouse = []
        for i in Z:
            if i < bed_find_1:
                bed_heights.append(i)
            elif bed_find_1 < i and i < bed_find_2:
                platform_1_mouse.append(i)
            elif i > bed_find_2:
                platform_2_mouse.append(i)

        bed = int(np.mean(np.array(bed_heights)))
        platform_1 = int(np.mean(np.array(platform_1_mouse)))
        platform_2 = int(np.mean(np.array(platform_2_mouse)))

        toothness = np.mean(self.voxels > 2000, axis = (0, 1)) # finds lowest z slice of teeth to avoid cutting them off.

        # plt.plot(toothness)
        # plt.title("Mean Density of xy Slices on z Axis")
        # plt.xlabel("Z")
        # plt.ylabel("Mean xy Slice Density")
        # plt.show()

        teeth_here = []
        lower_teeth = []
        mid_teeth = []
        upper_teeth = []
        voxel_arrays = []
        for i in range(0, toothness.shape[0]-1):
            if toothness[i] > 0:
                if i < 100:
                    lower_teeth.append(i)
                elif 125 < i < 215:
                    mid_teeth.append(i)
                elif 225 < i:
                    upper_teeth.append(i)


        if len(upper_teeth) != 0:
            upper_tooth_stop = upper_teeth[0]
        if len(lower_teeth) != 0:
            low_tooth_stop = lower_teeth[0]
        if len(mid_teeth) != 0:
            mid_tooth_stop = mid_teeth[0]




        #print(low_tooth_stop, mid_tooth_stop, upper_tooth_stop)
        if len(lower_teeth) != 0 and len(mid_teeth) != 0:
            self.voxels[:,:,:low_tooth_stop - 1][self.voxels[:,:,:low_tooth_stop - 1] < 2000] = -1000
            self.voxels[:,:,mid_tooth_stop-10:mid_tooth_stop-1][self.voxels[:,:,mid_tooth_stop-10:mid_tooth_stop-1] < 2000] = -1000
            if len(upper_teeth) != 0:
                self.voxels[:,:,upper_tooth_stop-10:upper_tooth_stop-1][self.voxels[:,:,upper_tooth_stop-10:upper_tooth_stop-1] < 2000] = -1000 #for the five mouse scans it was easy to add bed removal into this function since the scans are split along the beds anyways, so I did.
            else:
                self.voxels[:, :, platform_2-3:platform_2] = -1000

            Mouse = self.voxels[:,:,:]
            midline =int(Mouse.shape[0]/2)



            mouse_1 = Mouse[:midline, :, low_tooth_stop - 3:mid_tooth_stop-10]
            mouse_2 = Mouse[midline:, :, low_tooth_stop - 3:mid_tooth_stop-10]
            if len(upper_teeth) != 0:
                mouse_3 = Mouse[:midline,: , mid_tooth_stop-3:upper_tooth_stop - 10]
                mouse_4 = Mouse[midline:, :, mid_tooth_stop-3:upper_tooth_stop - 10]
                mouse_5 = Mouse[midline-mouse_wide:midline+mouse_wide,:, upper_tooth_stop - 3:]
            else:
                mouse_3 = Mouse[:midline,: , mid_tooth_stop-3:platform_2]
                mouse_4 = Mouse[midline:, :, mid_tooth_stop-3:platform_2]
                mouse_5 = Mouse[midline-mouse_wide:midline+mouse_wide,:, platform_2:]
        else:
            Mouse = self.voxels[:, :, :]
            midline = int(Mouse.shape[0]/2)
            mouse_1 = Mouse[:midline, :, bed:platform_1]
            mouse_2 = Mouse[midline:, :, bed:platform_1]
            mouse_3 = Mouse[:midline, :, platform_1:platform_2]
            mouse_4 = Mouse[midline:, :, platform_1:platform_2]
            mouse_5 = Mouse[midline-mouse_wide:midline+mouse_wide, :, platform_2:]
            #print('mice separated')
            #plt.imshow(np.max(mouse_1, axis = 1))

        final_arrays = [mouse_1, mouse_2, mouse_3, mouse_4, mouse_5]

        # for array in final_arrays:
        #     plt.imshow(array[:, 150, :], cmap = "bone")
        #     plt.show()

        if len(mouseIDs) < 5:
            final_mouse_IDs = []
            ID_index = 0
            for array in final_arrays:

                #voxelhelp.view_xyz_mips(array)
                #print(f'np.sum(array>-200): {np.sum(array>-200)}')
                if np.sum(array>-200) < 290317: # require at least 3mL of tissue-desity in scan
                    final_mouse_IDs.append('x')
                else:
                    final_mouse_IDs.append(mouseIDs[ID_index])
                    ID_index += 1
            if not ID_index==len(mouseIDs)-1:
                print(f'warning!! mouse IDs may not have been correctly mapped to voxel arrays')

        else:
            final_mouse_IDs = mouseIDs

        print(final_mouse_IDs)

        voxel_arrays.append(mouse_1)
        voxel_arrays.append(mouse_2)
        voxel_arrays.append(mouse_3)
        voxel_arrays.append(mouse_4)
        voxel_arrays.append(mouse_5)

        #print('arrays listed haha')



        one_mouse_scans = [OneMouseScan(voxels, self.voxel_spacing, name, self.date) for voxels, name in zip(final_arrays, final_mouse_IDs)]

        print('done splitting five-mouse scan')

        return one_mouse_scans


    def test_three_mouse(path):
        xradct_scan = xradct.XRadCT.from_dicom_files(path)
        split_scans = OneRowScan.to_individual_mice(xradct_scan)

    def test_five_mouse(path):
        xradct_scan = xradct.XRadCT.from_dicom_files(path)
        split_scans = FiveMouseScan.to_single_mice(xradct_scan)

