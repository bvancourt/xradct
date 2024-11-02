import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import os
import logging
##########################################

import voxelhelp
import xradct
import scan_processing
import head_segmentation
import tumor_analysis
#import plot_helper_excel as pltxl


class Controller():
    def __init__(self, task_data, base_path=''):
        self.task_data = task_data
        self.base_path = base_path

        self.false_list = [False, 'false', 'FALSE', 'False', 'false ', 'FALSE ', 'False ']
        self.true_list = [True, 'TRUE', 'true', 'True', 'TRUE ', 'true ', 'True ']

    def execute(self, wrapped_by_process_pool_excuter=False):
        tumor_vol = []
        date = []
        mouseID = []
        vol_out = []
        seg_out = []
        scan_out = []
        split_out = []
        output_output = []
        output_input_type = []
        problems = []
        exclusions = []
        warning_str = []
        plt.ion()
        for i, request in enumerate(self.task_data['scan_location']):

            try:
                # handle a request to find tumor volumes from scan
                if self.base_path=='':
                    path_to_scan = self.task_data['scan_location'][i]
                else:
                    path_to_scan = os.path.join(self.base_path, self.task_data['scan_location'][i])

                if self.task_data["override"][i] in self.true_list:
                    true_mouseIDs = self.task_data["true_mouseIDs"][i]
                    print(true_mouseIDs)
                if not os.path.exists(os.path.join(self.base_path, self.task_data['output_path'][0])):
                    os.makedirs(os.path.join(self.base_path, self.task_data['output_path'][0]))

                if self.task_data['input_type'][i] == 'Xrad one mouse':
                    if (path_to_scan[-4:] == '.nii')|(path_to_scan[-7:] == '.nii.gz'):
                        xradct_scan = xradct.XRadCT.from_nii(path_to_scan)
                    else:
                        xradct_scan = xradct.XRadCT.from_dicom_files(path_to_scan) #creates instance of XRadCT class object, and uses the from_dicom_files to obtain the voxels and name from folder

                    one_mouse_scan = scan_processing.OneMouseScan(xradct_scan.voxels, xradct_scan.voxel_spacing, xradct_scan.name, xradct_scan.date)
                    mouse_only_scan = scan_processing.OneMouseScan.clean_up_scan(one_mouse_scan, skip_sides = True, skip_fronts = True)

                    if self.task_data["override"][i] in self.true_list:
                        mouse_only_scan.name = true_mouseIDs
                    #creates instance of OneMouse scan object and calls both bed_removal and remove_xtra_mice
                    print(mouse_only_scan.empty)

                    if mouse_only_scan.empty == True:
                        print("This scan is empty")
                        scan_out.append("none")
                        seg_out.append("none")
                        output_input_type.append("none")
                        vol_out.append("none")
                        if self.task_data['override'][i] in self.true_list:
                            mouseID.append(true_mouseIDs[i])
                        else:
                            mouseID.append(mouse_only_scan[:2]+"M" +mouse_only_scan[-1:])
                        exclusions.append("none")
                        warning_str.append("none")
                        problems.append('empty Scan')
                    else:
                        if self.task_data['return_split_scan'][i] in self.true_list:
                            voxelhelp.write_nii(mouse_only_scan.voxels, f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}{self.task_data['dpi'][i]}")
                            split_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}")
                        else:
                            split_out.append("none")
                        if self.task_data['return_head_scan'][i] in self.true_list and self.task_data['return_head_seg'][i] in self.true_list:
                            try:
                                head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_seg_file = True, seg_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii", save_scan_file = True, scan_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                tumor_volume = segmentation_info["tumor_volume"]
                                exclude = segmentation_info["recommend_excluding"]
                                warnings = segmentation_info["warning_strings"]

                                seg_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                scan_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                output_input_type.append("Xrad one mouse")
                            except BaseException:
                                logging.exception("head_segmentation error :(")
                                tumor_volume = "calculation not available"
                                exclude = True
                                warnings = "segmentation unsuccessful"

                                plt.imshow(mouse_only_scan.voxels[:,90,:])
                                plt.pause(2)
                                plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")





                        elif self.task_data['return_head_scan'][i] in self.true_list and self.task_data['return_head_seg'][i] in self.false_list:
                            try:
                                head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_scan_file = True, scan_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                tumor_volume = segmentation_info["tumor_volume"]
                                exclude = segmentation_info["recommend_excluding"]
                                warnings = segmentation_info["warning_strings"]
                                scan_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                seg_out.append("none")

                                output_input_type.append("Xrad one mouse")
                            except BaseException:
                                logging.exception("head_segmentation error :(")
                                tumor_volume = "calculation not available"
                                exclude = True
                                warnings = "segmentation unsuccessful"
                                plt.imshow(mouse_only_scan.voxels[:,90,:])
                                plt.pause(2)
                                plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")



                        elif self.task_data['return_head_seg'][i] in self.true_list and self.task_data['return_head_scan'][i] in self.false_list:
                            try:
                                head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_seg_file = True, seg_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                tumor_volume = segmentation_info["tumor_volume"]
                                exclude = segmentation_info["recommend_excluding"]
                                warnings = segmentation_info["warning_strings"]
                                seg_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                scan_out.append("none")

                                output_input_type.append("none")
                            except BaseException:
                                logging.exception("head_segmentation error :(")
                                tumor_volume = "calculation not available"
                                exclude = True
                                warnings = "segmentation unsuccessful"

                                plt.imshow(mouse_only_scan.voxels[:,90,:])
                                plt.pause(2)
                                plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")


                        else:
                            try:
                                head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan)
                                seg_out.append("none")
                                scan_out.append("none")
                                tumor_volume = segmentation_info["tumor_volume"]
                                exclude = segmentation_info["recommend_excluding"]
                                warnings = segmentation_info["warning_strings"]

                                output_input_type.append("none")

                            except BaseException:
                                logging.exception("head_segmentation error :(")
                                tumor_volume = "calculation not available"
                                exclude = True
                                warnings = "segmentation unsuccessful"

                                plt.imshow(mouse_only_scan.voxels[:,90,:])
                                plt.pause(2)
                                plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")



                        if self.task_data["tumor_volume"][i] in self.true_list:
                            tumor_vol.append(tumor_volume) #takes in segmentation of head and use the voxel dimensions to return a volume in mm^3
                            vol_out.append(f"{self.task_data['output_path'][i]}\tumor_volumes.xlsx")
                            if tumor_volume < 10:
                                problems.append("Abnormally low tumor volume - inspect scan")
                        else:
                            tumor_vol.append("none")
                            vol_out.append("none")
                            exclusions.append("none")
                            warnings.append('none')
                            problems.append('none')


                        mouseID.append(mouse_only_scan.name)

                        warning_str.append(warnings)
                        exclusions.append(exclude)

                    date.append(self.task_data['dpi'][i])
                    output_output.append(self.task_data["output_path"][i])



#########################################################################################################################################################
                elif self.task_data['input_type'][i] == 'Xrad three mouse':
                    xradct_scan = xradct.XRadCT.from_dicom_files(path_to_scan)#creates instance of XRadCT class object, and uses the from_dicom_files to obtain the voxels and name from folder
                    split_scans = scan_processing.OneRowScan.to_individual_mice(xradct_scan)#returns an array of One-Mouse objects.
                    if not os.path.exists(os.path.join(self.base_path, self.task_data['output_path'][i])):
                        os.makedirs(os.path.join(self.base_path, self.task_data['output_path'][i]))
                    if self.task_data["override"][i] in self.true_list:
                        true_mouseIDs = [with_spaces.replace(' ', '') for with_spaces in self.task_data["true_mouseIDs"][i].split(',')]
                        print(true_mouseIDs)
                    override_index = -1
                    for j in split_scans:
                        override_index = override_index+1
                        mouse_only_scan = scan_processing.OneMouseScan.clean_up_scan(j, skip_sides = True, skip_fronts = True)
                        if self.task_data["override"][i] in self.true_list:
                            mouse_only_scan.name = true_mouseIDs[override_index]
                        #creates instance of OneMouse scan object and calls both bed_removal and remove_xtra_mice
                        #plt.imshow(mouse_only_scan.voxels[:,90,:])
                        #plt.title(mouse_only_scan.name)
                        #plt.show(block=False)
                        print(mouse_only_scan.empty)
                        #voxelhelp.view_xyz_mips(mouse_only_scan.voxels, annotation = [0, 0, 0])
                        if mouse_only_scan.empty == True:
                            print("This scan is empty")
                            scan_out.append("none")
                            seg_out.append("none")
                            split_out.append("none")
                            output_input_type.append("none")
                            vol_out.append("none")
                            mouseID.append(mouse_only_scan.name)
                            exclusions.append('none')
                            warning_str.append('none')
                            problems.append("Empty scan")

                            plt.imshow(mouse_only_scan.voxels[:,90,:])
                            plt.pause(2)
                            plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_susEmpty.png")
                        else:
                            if self.task_data['return_split_scan'][i] in self.true_list:
                                voxelhelp.write_nii(mouse_only_scan.voxels, f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}")
                                split_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}")
                            else:
                                split_out.append("none")
                            if self.task_data['return_head_scan'][i] in self.true_list and self.task_data['return_head_seg'][i] in self.true_list:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_seg_file = True, seg_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii", save_scan_file = True, scan_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    seg_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                    scan_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")

                                    output_input_type.append("Xrad one mouse")
                                except BaseException:
                                    logging.exception("head_segmentation error :(")
                                    tumor_volume = "calculation not available"
                                    exclude = True
                                    warnings = "segmentation unsuccessful"

                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")





                            elif self.task_data['return_head_scan'][i] in self.true_list and self.task_data['return_head_seg'][i] in self.false_list:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_scan_file = True, scan_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    scan_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    seg_out.append("none")

                                    output_input_type.append("Xrad one mouse")
                                except BaseException:
                                    logging.exception("head_segmentation error :(")
                                    tumor_volume = "calculation not available"
                                    exclude = True
                                    warnings = "segmentation unsuccessful"
                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")



                            elif self.task_data['return_head_seg'][i] in self.true_list and self.task_data['return_head_scan'][i] in self.false_list:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_seg_file = True, seg_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    seg_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                    scan_out.append("none")
                                    output_input_type.append("none")
                                except BaseException:
                                    logging.exception("head_segmentation error :(")
                                    tumor_volume = "calculation not available"
                                    exclude = True
                                    warnings = "segmentation unsuccessful"

                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")


                            else:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan)
                                    seg_out.append("none")
                                    scan_out.append("none")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    output_input_type.append("none")

                                except BaseException:
                                    logging.exception("head_segmentation error :(")
                                    tumor_volume = "calculation not available"
                                    exclude = True
                                    warnings = "segmentation unsuccessful"

                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_day{self.task_data['date'][i]} scan")


                            if self.task_data["tumor_volume"][i] in self.true_list:
                                tumor_vol.append(tumor_volume) #takes in segmentation of head and use the voxel dimensions to return a volume in mm^3
                                vol_out.append(f"{self.task_data['output_path'][i]}\tumor_volumes.xlsx")
                                if tumor_volume < 10:
                                    problems.append("Abnormally low tumor volume - inspect scan")
                            else:
                                tumor_vol.append("none")
                                vol_out.append("none")
                                exclusions.append("none")
                                warnings.append('none')
                                problems.append('none')


                            mouseID.append(mouse_only_scan.name)

                            warning_str.append(warnings)
                            exclusions.append(exclude)

                        date.append(self.task_data['dpi'][i])
                        output_output.append(self.task_data["output_path"][i])



################################################################################################################################
                elif self.task_data['input_type'][i] == 'Xrad five mouse':
                    xradct_scan = xradct.XRadCT.from_dicom_files(path_to_scan)#creates instance of XRadCT class object, and uses the from_dicom_files to obtain the voxels and name from folder
                    split_scans = scan_processing.FiveMouseScan.to_single_mice(xradct_scan)#returns an array of One-Mouse objects.
                    if not os.path.exists(os.path.join(self.base_path, self.task_data['output_path'][i])):
                        os.makedirs(os.path.join(self.base_path, self.task_data['output_path'][0]))
                    if self.task_data["override"][i] in self.true_list:
                        true_mouseIDs = [with_spaces.replace(' ', '') for with_spaces in self.task_data["true_mouseIDs"][i].split(',')]
                        print(true_mouseIDs)
                    override_index = -1

                    for j in split_scans:
                        override_index = override_index+1
                        #plt.imshow(j.voxels[:,50, :], cmap = 'bone')
                        #plt.show()
                        mouse_only_scan = scan_processing.OneMouseScan.clean_up_scan(j)#creates instance of OneMouse scan object and calls both bed_removal and remove_xtra_mice
                        if self.task_data["override"][i] in self.true_list:
                            mouse_only_scan.name = true_mouseIDs[override_index]
                        #voxelhelp.view_xyz_mips(mouse_only_scan.voxels, annotation = [0, 0, 0])

                        print(mouse_only_scan.empty)
                        if mouse_only_scan.empty == True:
                            print("This scan is empty")
                            scan_out.append("none")
                            seg_out.append("none")
                            split_out.append("none")
                            output_input_type.append("none")
                            vol_out.append("none")
                            mouseID.append(mouse_only_scan.name)
                            exclusions.append("none")
                            warning_str.append("none")
                            problems.append('Empty Scans')


                            plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_susEmpty.png")
                        else:
                            if self.task_data['return_split_scan'][i] in self.true_list:
                                voxelhelp.write_nii(mouse_only_scan.voxels, f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}")
                                split_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}")
                            else:
                                split_out.append("none")
                            if self.task_data['return_head_scan'][i] in self.true_list and self.task_data['return_head_seg'][i] in self.true_list:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_seg_file = True, seg_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii", save_scan_file = True, scan_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    seg_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                    scan_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    output_input_type.append("Xrad one mouse")
                                except BaseException:
                                    logging.exception("head_segmentation error :(")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]


                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")





                            elif self.task_data['return_head_scan'][i] in self.true_list and self.task_data['return_head_seg'][i] in self.false_list:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_scan_file = True, scan_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    scan_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_scan.nii")
                                    seg_out.append("none")
                                    output_input_type.append("Xrad one mouse")
                                except BaseException:
                                    logging.exception("head_segmentation error :(")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")



                            elif self.task_data['return_head_seg'][i] in self.true_list and self.task_data['return_head_scan'][i] in self.false_list:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan, save_seg_file = True, seg_file_path = f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    seg_out.append(f"{self.task_data['output_path'][i]}\{mouse_only_scan.name}_day{self.task_data['dpi'][i]}_head_seg.nii")
                                    scan_out.append("none")
                                    output_input_type.append("none")
                                except BaseException:
                                    logging.exception("head_segmentation error :(")#main-type function that calls all other functions in module to segment head
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]

                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")


                            else:
                                try:
                                    head_only_seg, head_only_scan, segmentation_info = head_segmentation.final_head(mouse_only_scan)
                                    seg_out.append("none")
                                    scan_out.append("none")
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]
                                    output_input_type.append("none")

                                except BaseException:
                                    logging.exception("head_segmentation error :(") #main-type function that calls all other functions in module to segment head
                                    tumor_volume = segmentation_info["tumor_volume"]
                                    exclude = segmentation_info["recommend_excluding"]
                                    warnings = segmentation_info["warning_strings"]

                                    plt.imshow(mouse_only_scan.voxels[:,90,:])
                                    plt.pause(2)
                                    plt.savefig(f"{os.path.join(self.base_path, self.task_data['output_path'][0])}\{mouse_only_scan.name}_segFailure.png")
                                    problems.append(f"Segmentation failed for {mouse_only_scan.name}_{self.task_data['date'][i]} scan")


                            if self.task_data["tumor_volume"][i] in self.true_list:
                                tumor_vol.append(tumor_volume) #takes in segmentation of head and use the voxel dimensions to return a volume in mm^3
                                vol_out.append(f"{self.task_data['output_path'][i]}\tumor_volumes.xlsx")
                                if tumor_volume < 10:
                                    problems.append("Abnormally low tumor volume - inspect scan")
                            else:
                                tumor_vol.append("none")
                                vol_out.append("none")


                            mouseID.append(mouse_only_scan.name)




                            warning_str.append(warnings)
                            exclusions.append(exclude)

                        date.append(self.task_data['dpi'][i])
                        output_output.append(self.task_data["output_path"][i])


                output_file_list = list(zip(mouseID, seg_out, scan_out, split_out))
                output = self.task_data['output_path'][0]
                #output_data = pd.DataFrame((output_file_list), columns = ["Mouse ID", "Segmentation Location", "Scan Location"])
                vol_data_output = list(zip(mouseID, date, tumor_vol, exclusions, seg_out, scan_out, split_out))
                tumor_data = pd.DataFrame((vol_data_output), columns = ["Mouse ID", "Days Post Implantation", "Tumor Volume (mm^3)", "Exclude Recommended?", "Segmentation Location", "Scan Location", "Split Scan Location"])

                one_mouse_output = list(zip(mouseID, date, output_input_type, scan_out, output_output))
                #one_mouse_info = pd.DataFrame((one_mouse_output), columns = ["Mouse ID", "Days Post Implantation", "input_type", "scan_location", "output_path"])
                problems_array = list(zip(mouseID, problems, exclusions, warning_str))
                problems_data = pd.DataFrame((problems_array), columns = ["Mouse ID", "Problem Log", "Excluded?", "Warning Strings"])

                if not wrapped_by_process_pool_excuter:
                    print(os.path.join(self.base_path, self.task_data['output_path'][0]))

                    #output_data.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'output_files.xlsx'))

                    problems_data.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'problem_log.xlsx'))
                    print(os.path.join(self.base_path, self.task_data['output_path'][0], 'tumor_volumes.xlsx'))
                    tumor_data.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'tumor_volumes.xlsx'))

                    #one_mouse_info.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'one-mouse-scans.xlsx'))
                    #print(os.path.join(self.base_path, self.task_data['output_path'][0], 'one-mouse-scans.xlsx'))
                else:
                    dfs_out = []
                    paths_out = []

                    dfs_out.append(tumor_data)
                    paths_out.append(os.path.join(self.base_path, self.task_data['output_path'][0], 'tumor_volumes.xlsx'))

                    #dfs_out.append(output_data)
                    #paths_out.append(os.path.join(self.base_path, self.task_data['output_path'][0], 'output_files.xlsx'))

                    dfs_out.append(problems_data)
                    paths_out.append(os.path.join(self.base_path, self.task_data['output_path'][0], 'problem_log.xlsx'))


                    problems_data = pd.DataFrame(np.array(problems), columns = ["Problem Log"])
                    problems_data.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'problem_log.xlsx'))

                    #dfs_out.append(one_mouse_info)
                    #paths_out.append(os.path.join(self.base_path, self.task_data['output_path'][0], 'one-mouse-scans.xlsx'))


                    return dfs_out, paths_out


                #plotxl.plot_from_autocontour_output(os.path.join(self.task_data['output_path'][1], 'tumor_volumes.xlsx'), self.task_data['output_path'][1])

            except BaseException:
                logging.exception("task_controller has run into an error :(")
                problems.append(f"Scan not processed for {self.task_data['scan_location'][i]}")

                problems_data = pd.DataFrame(np.array(problems), columns = ["Problem Log"])
                problems_data.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'problem_log.xlsx'))

                if not wrapped_by_process_pool_excuter:
                    problems_data.to_excel(os.path.join(self.base_path, self.task_data['output_path'][0], 'problem_log.xlsx'))
                else:
                    return [], []
