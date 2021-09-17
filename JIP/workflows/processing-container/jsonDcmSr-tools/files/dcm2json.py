import sys, os
import math
import json
import os
import subprocess
import glob
import xml.etree.ElementTree as ET


batch_folders = [f for f in glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], '*'))]
for batch_element_dir in batch_folders:
    
    element_input_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_IN_DIR'])    
    if os.path.exists(element_input_dir):

        element_output_dir = os.path.join(batch_element_dir, os.environ['OPERATOR_OUT_DIR'])
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)
            
        element_output_json = os.path.join(element_output_dir, os.environ['OUTPUT_JSON_FILENAME'])
        element_temp_xml = os.path.join(element_output_dir, "temp.xml")
        
        print(f"Out dir {element_output_dir}")
        print(f"Out file {element_output_json}")

        dcm_files = glob.glob(os.path.join(element_input_dir,"*.dcm"))
        if len(dcm_files) > 0:
            dcm_file = dcm_files[0]
            print(f"Converting {dcm_file} to {element_output_json}")
        else: 
            print(f"No Dicom files found in {element_input_dir}.")
            sys.exit()
            
        try:        
            c = [f"dsr2xml", dcm_file, element_temp_xml]
            print(f"running command: {c}")
            resp = subprocess.check_output([
                f"dsr2xml",
                dcm_file, element_temp_xml
            ], stderr=subprocess.STDOUT)
            
            root = ET.parse(element_temp_xml).getroot()
            found_data = root.find('./document/content/container/text/value')
            if found_data != None:
                json_data_string = found_data.text                
                json_data = json.loads(json_data_string)            
                with open(element_output_json, 'w') as outfile:
                    json.dump(json_data, outfile)            
                os.remove(element_temp_xml)
            else:
                raise ValueError("XML does not contain metrics data in './document/content/container/text/value'")
            
        except subprocess.CalledProcessError as e:
            print("DCMTK Error: ")
            print(e.output)

    
