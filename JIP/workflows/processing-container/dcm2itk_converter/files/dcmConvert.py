import sys, os
import glob
import json
import subprocess
# transform imgs:
import dicom2nifti
# read dicom:
import pydicom
import SimpleITK as sitk
import numpy as np
import pydicom_seg

DCMQI = '/dcmqi/dcmqi-1.2.2-linux/bin/'
PLASTIMATCH = '/dcmqi/dcmqi-1.2.2-linux/bin/'

# For local testng

# os.environ["WORKFLOW_DIR"] = "<your data directory>"
# os.environ["BATCH_NAME"] = "batch"
# os.environ["OPERATOR_IN_DIR"] = "initial-input"
# os.environ["OPERATOR_OUT_DIR"] = "output"
# os.environ["OUTPUT_FORMAT"] = "nii"

# From the template
batch_folders = [f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ["OPERATOR_IN_DIR"], '*'))]

for batch_element_dir in batch_folders:
    if os.path.isdir(batch_element_dir):
        #================
        # convert images
        #================    
        inst_dir_name = os.path.basename(os.path.normpath(batch_element_dir))
        img_element_input_dir = os.path.join(batch_element_dir, 'img')
        img_element_output_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['OPERATOR_OUT_DIR'], inst_dir_name, 'img')
        img_element_output_file = os.path.join(img_element_output_dir, f"img.{os.environ['OUTPUT_FORMAT']}")
        print(f'Converting DICOM Image: {img_element_input_dir} to {img_element_output_dir}')
        
        if len(os.listdir(img_element_input_dir)) == 0:
            print(f"No DICOM images found in directory {img_element_input_dir}")
        else:
            try:
                plastimatch_command = [f"plastimatch convert --input {img_element_input_dir} --output-img {img_element_output_file}"]
                print('Executing', " ".join(plastimatch_command))
                resp = subprocess.check_output(plastimatch_command, stderr=subprocess.STDOUT, shell=True)
                print(resp)
            except subprocess.CalledProcessError as e:
                print("Error with plastimatch. ", e.output)
                print("Abort !")
                exit(1) 
        
        #=======================
        # convert segmentations
        #=======================
        seg_element_input_dir = os.path.join(batch_element_dir, 'seg')
        seg_element_output_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['OPERATOR_OUT_DIR'], inst_dir_name, 'seg')
        
        if not os.path.exists(seg_element_input_dir):
            print(f"No DICOM segmentations found in directory {seg_element_input_dir}")
        else:    
            if not os.path.exists(seg_element_output_dir):
                os.makedirs(seg_element_output_dir)

            dcm_paths = glob.glob(f'{seg_element_input_dir}/*.dcm')    
            
            for i, dcm_filepath in enumerate(dcm_paths):
                ds = pydicom.dcmread(dcm_filepath)
                modality = ds.Modality
                out_file_lm = os.path.join(seg_element_output_dir, f"{(i+1):03d}.{os.environ['OUTPUT_FORMAT']}")
                print(f'Converting DICOM-{modality} segmentation: {dcm_filepath} to {seg_element_output_dir}')
                
                #-----------------------
                # convert dicom-rtstuct # MAY NOT BE REQUIRED
                #-----------------------
                if modality.lower() == "rtstruct":
                    try: 
                        plastimatch_command = [f"plastimatch convert --input {dcm_filepath} --output-labelmap {out_file_lm} --referenced-ct {img_element_input_dir}"]  
                        print('Executing', " ".join(plastimatch_command))
                        resp = subprocess.check_output(plastimatch_command, stderr=subprocess.STDOUT, shell=True)
                        print(resp)
                    except subprocess.CalledProcessError as e:
                        print("Error with plastimatch. ", e.output)
                        print("Abort !")
                        exit(1) 
                        
                #-------------------
                # convert dicom-seg
                #-------------------
                elif modality.lower() == "seg":
                    dcm = pydicom.dcmread(dcm_filepath)
                    reader = pydicom_seg.MultiClassReader()
                    result = reader.read(dcm)
                    out_image = result.image 
                    output_dir = os.path.join(seg_element_output_dir, out_file_lm)
                    sitk.WriteImage(out_image, output_dir, True)      
                    
                    
                else:
                    raise TypeError(f"DICOM modality must be either 'RTSTRUCT' or 'SEG'. Given modality: {modality}")
