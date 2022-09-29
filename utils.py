# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:50:11 2022

@author: lowes
"""
import os
import glob
import numpy as np
import SimpleITK as sitk
import pydicom as dcm

def find_all_files(file_path):
    print(f'reading file list in {file_path}')
    unsorted_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # the next line can be used if you need to filter files
            # if ".dcm" in file:  # exclude non-dicoms, good for messy folders
            unsorted_list.append(os.path.join(root, file))
    print(f'{len(unsorted_list)} files found.')
    return unsorted_list

def read_dicom_files_carefully(file_path):
    slices = []
    unsorted_list = find_all_files(file_path)
    if len(unsorted_list) < 1:
        print("Did not find any files")
        return None

    for dicom_loc in unsorted_list:
        valid_file = True
        ds = None
        try:
            ds = dcm.read_file(dicom_loc, force=False)
        except dcm.errors.InvalidDicomError:
            print('Exception when trying to read and parse', dicom_loc)
            valid_file = False
        except Exception as ex:
            print('Exception when trying to read and parse', dicom_loc)
            template = "An exception of type {0} occurred. Arguments: {1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            valid_file = False
        if valid_file:
            # uncompress files (using the gdcm package)
            try:
                ds.decompress()
            except:
                print('an instance in file %s" could not be decompressed.' % dicom_loc)
                valid_file = False
            if valid_file:
                slices.append(ds)

    if len(slices) < 2:
        print("Error: Not enough slices in scan")
        return None

    slices.sort(key=lambda x: int(x.InstanceNumber))
    full_scan = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    full_scan = full_scan.astype(np.int16)
    # num, x, y= full_scan.shape

    # image_data = vtk.vtkImageData()
    # depth_array = numpy_to_vtk(num_array=full_scan.ravel(), deep=True)
    # image_data.SetDimensions((x, y, num))
    # image_data.SetSpacing([1,1,1])
    # image_data.SetOrigin([0,0,0])
    # image_data.GetPointData().SetScalars(depth_array)
    return full_scan