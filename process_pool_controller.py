import concurrent.futures
import os
import pandas as pd
import logging
import time

import voxelhelp
import xradct
import head_segmentation
import scan_processing
import task_controller


class RowPerMouseController:
    def __init__(self, control_file_path, paths_relative_to_xlsx=True):
        self.control_file_path = control_file_path
        self.paths_relative_to_xlsx = paths_relative_to_xlsx
        if paths_relative_to_xlsx:
            self.base_path = os.path.split(self.control_file_path)[0]

        self.task_data = pd.read_excel(self.control_file_path)

        # Organize requests by source scan (each source scan can be alayzed in a different process)
        self.source_paths = []
        self.requests_by_source = []
        #self.input_types = [] # Didn't end up using this.

        self.false_list = [False, 'false', 'FALSE', 'False', 'false ', 'FALSE ', 'False ']
        self.true_list = [True, 'TRUE', 'true', 'True', 'TRUE ', 'true ', 'True ']

        # note: i and j belowboth loop over rows the same, but outer loop only does anything the first time it encounters a given source path.
        for i in range(self.task_data.shape[0]):
            if not self.task_data['rel_input_path'][i] in self.source_paths:
                requests_from_this_source = []
                for j in range(self.task_data.shape[0]):
                    if self.task_data['rel_input_path'][i]==self.task_data['rel_input_path'][j]:
                        if ((self.task_data.iloc[j]['find_tumor_volume'] in self.true_list) or
                            (self.task_data.iloc[j]['save_head_seg'] in self.true_list) or
                            (self.task_data.iloc[j]['save_head_scan'] in self.true_list) or
                            (self.task_data.iloc[j]['save_split_scan'] in self.true_list)): # No need to add requests for nothing to the list
                            requests_from_this_source.append(self.task_data.iloc[j])
                if len(requests_from_this_source) > 0:
                    self.source_paths.append(self.task_data['rel_input_path'][i])
                    #self.input_types.append(self.task_data['input_type'][i])
                    self.requests_by_source.append(requests_from_this_source)

    def execute_requests_from_source(self, source_id):
        process_info = {
            'source_number' : source_id,
            'source_path' : self.source_paths[source_id],
            'request_info_dicts' : [],
            'error_strings' : []
        }

        try:
            # Make sure all requests classify the source as having the same input type.
            if len(set([request['input_type'] for request in self.requests_by_source[source_id]])) > 1:
                error_string = f"Source scan at {self.source_paths[source_id]} was not split, because it was multiple input types, {set([request['input_type'] for request in self.requests_by_source])}, were provided."
                process_info['error_strings'].append(error_string)
                return process_info

            if self.paths_relative_to_xlsx:
                full_source_path = os.path.join(self.base_path, self.source_paths[source_id])
            else:
                full_source_path = self.source_paths[source_id]

            # Read source scan
            try:
                if (full_source_path[-4:] == '.nii')|(full_source_path[-7:] == '.nii.gz'):
                    source_scan = xradct.XRadCT.from_nii(full_source_path)
                else:
                    source_scan = xradct.XRadCT.from_dicom_files(full_source_path)
                source_scan.nudge_CT_numbers()
            except BaseException:
                error_string = f"Source scan at {full_source_path} not split; error in xradct.XRadCT.from_dicom_files() or xradct.XRadCT.from_nii()."
                process_info['error_strings'].append(error_string)
                logging.exception(error_string)
                return process_info

            # Split source to one mouse scans
            try:
                if self.requests_by_source[source_id][0]['input_type'] == 'Xrad one mouse':
                    split_scans = [scan_processing.OneMouseScan(source_scan.voxels, source_scan.voxel_spacing, source_scan.name, source_scan.date)]
                elif self.requests_by_source[source_id][0]['input_type'] == 'Xrad one row':
                    split_scans = scan_processing.OneRowScan.to_individual_mice(source_scan)
                elif self.requests_by_source[source_id][0]['input_type'] == 'Xrad five mouse':
                    split_scans = scan_processing.FiveMouseScan.to_single_mice(source_scan)
                    if not len(split_scans) == 5:
                        process_info['error_strings'].append(f'scan_processing.FiveMouseScan.to_single_mice() returned {len(split_scans)} length list.')
            except BaseException:
                error_string = f"Source scan at {full_source_path} not split; error in scan_processing.[OneRowScan or ].to_individual_mice()."
                process_info['error_strings'].append(error_string)
                logging.exception(error_string)
                return process_info

            # Loop over requests (which correspond to rows of the controller spreadsheet).
            for request in self.requests_by_source[source_id]:
                request_info = {
                    'mouse_ID' : str(request['mouse_ID_override']),
                    'tumor_volume' : 'n/a',
                    'seg_save_path' : 'n/a',
                    'headscan_save_path' : 'n/a',
                    'split_scan_save_path' : 'n/a',
                    'source_number' : process_info['source_number'],
                    'error_strings' : [],
                    'warning_strings' : [],
                    'rel_output_path': request['rel_output_path'],
                    'mouse_position_index': int(request['mouse_position_index']),
                    'dpi' : int(request['dpi']),
                    'date' : request['date'],
                    'implant_date' : request['implant_date'],
                    'experiment_ID' : request['experiment_ID'],
                    'source_scan_path' : full_source_path
                }
                no_mouse = True
                if (int(request['mouse_position_index'])<len(split_scans)): # if there is no split scan to process
                    if not (type(split_scans[int(request['mouse_position_index'])]) == str):
                        no_mouse = False
                if no_mouse:
                    error_string = f"Request for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} failed, because no mouse was found at this position."
                    process_info['error_strings'].append(error_string)
                    request_info['error_strings'].append(error_string)
                    if request_info['mouse_ID'] == False:
                        request_info['mouse_ID'] = 'n/a'
                    process_info['request_info_dicts'].append(request_info)
                    return process_info
                else:
                    # Clean up scan
                    try:
                        mouse_only_scan = scan_processing.OneMouseScan.clean_up_scan(split_scans[int(request['mouse_position_index'])])
                        if type(split_scans[int(request['mouse_position_index'])]) == str:
                            error_string = f"Request for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} failed, because no mouse was found at this position."
                            process_info['error_strings'].append(error_string)
                            request_info['error_strings'].append(error_string)
                            if request_info['mouse_ID'] == False:
                                request_info['mouse_ID'] = 'n/a'
                            process_info['request_info_dicts'].append(request_info)
                            return process_info

                    except BaseException:
                        error_string = f"Request for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} failed due to error in scan_processing.OneMouseScan.clean_up_scan()."
                        process_info['error_strings'].append(error_string)
                        request_info['error_strings'].append(error_string)
                        logging.exception(error_string)
                        return process_info

                    # Determine mouse ID
                    if request_info['mouse_ID'] in self.false_list: # else it will already be correct
                        if request['input_type'] == 'Xrad one mouse':
                            request_info['mouse_ID'] = mouse_only_scan.name
                        elif request['input_type'] == 'Xrad one row':
                            request_info['mouse_ID'] = mouse_only_scan.name[:2]+"M" +mouse_only_scan.name[-1:]
                        elif request['input_type'] == 'Xrad five mouse':
                            request_info['mouse_ID'] = mouse_only_scan.name
                        else:
                            request_info['mouse_ID'] = 'n/a'

                # Save split scan, if necessary
                if request['save_split_scan'] in self.true_list:
                    try:
                        file_name = f"mouse_{request_info['mouse_ID']}_at_{int(request['dpi'])}dpi_split_scan.nii"
                        if self.paths_relative_to_xlsx:
                            ouput_base_path = os.path.join(self.base_path, request['rel_output_path'])
                        else:
                            ouput_base_path =request['rel_output_path']
                        if not os.path.exists(ouput_base_path):
                            os.makedirs(ouput_base_path)
                        split_scan_write_path = os.path.join(ouput_base_path, file_name)

                        if os.path.exists(split_scan_write_path) and not request['overwrite_existing']:
                            for incr in range(1, 100):
                                candidate_bath = split_scan_write_path[:-4]+f' ({str(incr)}).nii'
                                if not os.path.exists(candidate_bath):
                                    split_scan_write_path = candidate_bath
                                    break

                        request_info['split_scan_save_path'] = split_scan_write_path
                        voxelhelp.write_nii(mouse_only_scan.voxels, request_info['split_scan_save_path'])

                    except BaseException:
                        error_string = f"Failed to save split scan for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} for some reason."
                        process_info['error_strings'].append(error_string)
                        request_info['error_strings'].append(error_string)
                        logging.exception(error_string)

                # Run head segmentation, if necessary
                if (request['find_tumor_volume'] in self.true_list) or (request['save_head_seg'] in self.true_list) or (request['save_head_scan'] in self.true_list):
                    try:
                        request_info['force_use_head_only'] = (request['force_use_head_only'] in self.true_list)
                        head_only_seg, split_scan_seg, headscan, segmentation_info = head_segmentation.final_head(mouse_only_scan, force_use_head_only=request_info['force_use_head_only'])
                        request_info['tumor_volume'] = segmentation_info['tumor_volume']
                        request_info['deformation_factor'] = segmentation_info['deformation_factor']
                        request_info['error_strings'] += segmentation_info['error_strings']
                        request_info['warning_strings'] += segmentation_info['tooth_seg_info']['warning_strings']
                        request_info['warning_strings'] += segmentation_info['bone_seg_info']['warning_strings']
                        request_info['warning_strings'] += segmentation_info['resample_info']['warning_strings']
                        request_info['warning_strings'] += segmentation_info['warning_strings']
                        request_info['threshold_for_teeth'] = segmentation_info['tooth_seg_info']['teeth_thresh']
                        request_info['threshold_for_bone'] = segmentation_info['bone_seg_info']['bone_thresh']
                        request_info['bone_LR_map_error'] = segmentation_info['bone_LR_map_error']
                        request_info['recommend_excluding'] = segmentation_info['recommend_excluding']

                        if request['save_head_seg'] in self.true_list:
                            try:
                                file_name = f"mouse_{request_info['mouse_ID']}_at_{int(request['dpi'])}dpi_headscan_segmentation.nii"
                                if self.paths_relative_to_xlsx:
                                    ouput_base_path = os.path.join(self.base_path, request['rel_output_path'])
                                else:
                                    ouput_base_path =request['rel_output_path']
                                if not os.path.exists(ouput_base_path):
                                    os.makedirs(ouput_base_path)
                                head_seg_write_path = os.path.join(ouput_base_path, file_name)
                                if os.path.exists(head_seg_write_path) and not request['overwrite_existing']:
                                    for incr in range(1, 100):
                                        candidate_bath = head_seg_write_path[:-4]+f' ({str(incr)}).nii'
                                        if not os.path.exists(candidate_bath):
                                            head_seg_write_path = candidate_bath
                                            break

                                request_info['head_seg_save_path'] = head_seg_write_path
                                voxelhelp.write_nii(head_only_seg.int_image, request_info['head_seg_save_path'])
                            except BaseException:
                                error_string = f"Failed to save head segmentation for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} for some reason."
                                process_info['error_strings'].append(error_string)
                                request_info['error_strings'].append(error_string)
                                logging.exception(error_string)

                        if request['save_split_scan_seg']in self.true_list:
                            try:
                                file_name = f"mouse_{request_info['mouse_ID']}_at_{int(request['dpi'])}dpi_segmentation.nii"
                                if self.paths_relative_to_xlsx:
                                    ouput_base_path = os.path.join(self.base_path, request['rel_output_path'])
                                else:
                                    ouput_base_path =request['rel_output_path']
                                if not os.path.exists(ouput_base_path):
                                    os.makedirs(ouput_base_path)
                                split_seg_write_path = os.path.join(ouput_base_path, file_name)
                                if os.path.exists(head_seg_write_path) and not request['overwrite_existing']:
                                    for incr in range(1, 100):
                                        candidate_bath = split_seg_write_path[:-4]+f' ({str(incr)}).nii'
                                        if not os.path.exists(candidate_bath):
                                            split_seg_write_path = candidate_bath
                                            break

                                request_info['split_seg_save_path'] = split_seg_write_path
                                voxelhelp.write_nii(split_scan_seg.int_image, request_info['split_seg_save_path'])
                            except BaseException:
                                error_string = f"Failed to save split scan segmentation for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} for some reason."
                                process_info['error_strings'].append(error_string)
                                request_info['error_strings'].append(error_string)
                                logging.exception(error_string)

                        if request['save_head_scan'] in self.true_list:
                            try:
                                file_name = f"mouse_{request_info['mouse_ID']}_at_{int(request['dpi'])}dpi_headscan.nii"
                                if self.paths_relative_to_xlsx:
                                    ouput_base_path = os.path.join(self.base_path, request['rel_output_path'])
                                else:
                                    ouput_base_path =request['rel_output_path']
                                if not os.path.exists(ouput_base_path):
                                    os.makedirs(ouput_base_path)
                                headscan_write_path = os.path.join(ouput_base_path, file_name)
                                if os.path.exists(headscan_write_path) and not request['overwrite_existing']:
                                    for incr in range(1, 100):
                                        candidate_bath = headscan_write_path[:-4]+f' ({str(incr)}).nii'
                                        if not os.path.exists(candidate_bath):
                                            headscan_write_path = candidate_bath
                                            break

                                request_info['headscan_save_path'] = headscan_write_path
                                voxelhelp.write_nii(headscan.voxels, request_info['headscan_save_path'])
                            except BaseException:
                                error_string = f"Failed to save headscan for position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']} for some reason."
                                process_info['error_strings'].append(error_string)
                                request_info['error_strings'].append(error_string)
                                logging.exception(error_string)

                    except BaseException:
                        error_string = f"Failed to segment mouse position {int(request['mouse_position_index'])} of scan at {request['rel_input_path']}. Error in head_segmentation.final_head()."
                        process_info['error_strings'].append(error_string)
                        request_info['error_strings'].append(error_string)
                        logging.exception(error_string)

                process_info['request_info_dicts'].append(request_info)

        except BaseException:
            error_string = f'unhandled error in execute_requests_from_source({source_id})'
            process_info['error_strings'].append(error_string)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(process_info['source_path'])
            print(process_info['error_strings'])
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            logging.exception(error_string)

        return process_info

    def run(self, print_process_info=False):
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.execute_requests_from_source, i) for i in range(len(self.source_paths))]

            process_info_dicts = []
            df_rows = {} # output_path,
            for future in concurrent.futures.as_completed(futures):
                process_info_dicts.append(future.result())
                try:
                    # This stuff is for saving speadsheets of the request info (including tumor volumes and files written).
                    for ri_dict in process_info_dicts[-1]['request_info_dicts']:
                        new_df_row = [] # [(column, value), (column, value), ... ]
                        for ri_key in ri_dict.keys():
                            new_df_row.append((ri_key, ri_dict[ri_key]))

                        # determine location to save data from this region info dictionary
                        if self.paths_relative_to_xlsx:
                            ouput_base_path = os.path.join(self.base_path, ri_dict['rel_output_path'])
                        else:
                            ouput_base_path = ri_dict['rel_output_path']
                        if not os.path.exists(ouput_base_path):
                            os.makedirs(ouput_base_path)

                        if not ouput_base_path in df_rows.keys():
                            df_rows[ouput_base_path] = [new_df_row]
                        else:
                            df_rows[ouput_base_path].append(new_df_row)
                except BaseException:
                    error_string = f'Failed to prepare a process info dictionary for saving to xlsx:\n{process_info_dicts[-1]}'
                    logging.exception(error_string)

            for ouput_base_path in df_rows.keys():
                # Sorting data to put in the output spreadsheet. There must be a simpler way to do this...
                tuple_rows = df_rows[ouput_base_path]
                all_col_names = []
                for row in tuple_rows:
                    all_col_names += [tup[0] for tup in row]
                col_names_set = set(all_col_names)
                col_names = ['mouse_ID', 'dpi', 'tumor_volume', 'recommend_excluding', 'deformation_factor', 'bone_LR_map_error',
                             'error_strings', 'warning_strings', 'threshold_for_bone', 'threshold_for_teeth'
                ]
                for name in col_names_set:
                    if not name in col_names:
                        col_names.append(name)

                column_dict = {}
                for j, name in enumerate(col_names):
                    column_dict[name] = []
                    for i, row in enumerate(tuple_rows):
                        item_found = False
                        for item in row:
                            if item[0]==name:
                                item_found = True
                                column_dict[name].append(item[1])
                                break
                        if not item_found:
                            column_dict[name].append('')

                task_data_write_path = os.path.join(ouput_base_path, 'auto_seg_task_data.xlsx')
                #increment file name to avoid overwriting
                if os.path.exists(task_data_write_path):
                    for incr in range(1, 100):
                        candidate_bath = task_data_write_path[:-5]+f' ({str(incr)}).xlsx'
                        if not os.path.exists(candidate_bath):
                            task_data_write_path = candidate_bath
                            break

                df = pd.DataFrame(column_dict)
                df.to_excel(task_data_write_path)


        if print_process_info:
            print('-------------------------------------------------------------------------------------------')

            for pi_dict in process_info_dicts:
                print('Process Info:')
                print(pi_dict)
                #for ri_dict in pi_dict['request_info_dicts']:
                #    print('*                   *                       *                        *                        *')
                #    print(ri_dict)
                print('. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .')


class SplitPathController(RowPerMouseController):
    def __init__(self, control_file_path):
        self.control_file_path = control_file_path
        self.paths_relative_to_xlsx = False

        self.task_data = pd.read_excel(self.control_file_path)

        # Organize requests by source scan (each source scan can be alayzed in a different process)
        self.source_paths = []
        self.requests_by_source = []
        self.request_indices_by_source = []
        #self.input_types = [] # Didn't end up using this.

        self.false_list = [False, 'false', 'FALSE', 'False', 'false ', 'FALSE ', 'False ']
        self.true_list = [True, 'TRUE', 'true', 'True', 'TRUE ', 'true ', 'True ']

        # note: i and j below both loop over rows the same, but outer loop only does anything the first time it encounters a given source path.
        rel_input_paths = []
        rel_output_paths = []
        for i in range(self.task_data.shape[0]):
            this_source_path = os.path.join(self.task_data['source_path_part_1'][i], self.task_data['source_path_part_2'][i])
            this_output_path = os.path.join(self.task_data['output_path_part_1'][i], self.task_data['output_path_part_2'][i])
            rel_input_paths.append(this_source_path)
            rel_output_paths.append(this_output_path)

            if not this_source_path in self.source_paths:
                request_indices_from_this_source = []
                for j in range(self.task_data.shape[0]):
                    if os.path.join(self.task_data['source_path_part_1'][j], self.task_data['source_path_part_2'][j]) == this_source_path:
                        if ((self.task_data.iloc[j]['find_tumor_volume'] in self.true_list) or
                            (self.task_data.iloc[j]['save_head_seg'] in self.true_list) or
                            (self.task_data.iloc[j]['save_head_scan'] in self.true_list) or
                            (self.task_data.iloc[j]['save_split_scan'] in self.true_list)): # No need to add requests for nothing to the list
                            request_indices_from_this_source.append(j)

                if len(request_indices_from_this_source) > 0:
                    self.source_paths.append(this_source_path)
                    #self.input_types.append(self.task_data['input_type'][i])
                    self.request_indices_by_source.append(request_indices_from_this_source)

        self.task_data['rel_input_path'] = rel_input_paths
        self.task_data['rel_output_path'] = rel_output_paths

        self.requests_by_source = []
        for i, source_path in enumerate(self.source_paths):
            requests_from_this_source = [self.task_data.iloc[j] for j in self.request_indices_by_source[i]]
            self.requests_by_source.append(requests_from_this_source)


class RowPerSourceController:
    def __init__(self, task_data, base_path=''):
        self.task_data = task_data
        self.base_path = base_path

        self.true_list = [True, 'TRUE', 'true', 'True', 'TRUE ', 'true ', 'True ']

    def spawn_task_controller(self, task_row):
        print(task_row)
        ext_controller = task_controller.Controller(pd.DataFrame(dict(task_row), index=[0]), base_path=self.base_path)
        dfs_out, out_paths = ext_controller.execute(wrapped_by_process_pool_excuter=True)
        return dfs_out, out_paths

    def execute(self):
        task_rows = filter(
            lambda row: (row['tumor_volume'] in self.true_list) or (row['return_split_scan'] in self.true_list) or (row['return_head_scan'] in self.true_list) or (row['return_head_seg'] in self.true_list),
            [self.task_data.iloc[i] for i in range(self.task_data.shape[0])])
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.spawn_task_controller, task_row) for task_row in task_rows]

            task_data_by_out_path = {}
            for future in concurrent.futures.as_completed(futures):
                these_task_dfs, these_out_paths = future.result()
                for this_task_df, this_out_path in zip(these_task_dfs, these_out_paths):
                    if not this_out_path in task_data_by_out_path.keys():
                        task_data_by_out_path[this_out_path] = [this_task_df]
                    else:
                        task_data_by_out_path[this_out_path].append(this_task_df)

            for output_path in task_data_by_out_path.keys():
                print(output_path)
                final_df = pd.concat(task_data_by_out_path[output_path])

                final_df.to_excel(output_path)
