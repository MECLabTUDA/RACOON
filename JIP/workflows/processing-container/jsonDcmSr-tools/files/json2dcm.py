import sys, os
import math
import json
import os
from datetime import datetime
import subprocess
from jinja2 import Environment, FileSystemLoader, select_autoescape
import glob
import pydicom

#element_output_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['OPERATOR_OUT_DIR'])
#element_output_xml = os.path.join(element_output_dir, 'data.xml')
all_qm_jsons = glob.glob(os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['OPERATOR_IN_DIR'], '*.json'))
dir_cntr = 1

print(f"QM files {all_qm_jsons}")
    

for qm_json in all_qm_jsons:
    if os.path.isfile(qm_json):

        with open(qm_json) as json_file:
            qm_data_json = json.load(json_file)
        
        # Level 'batch' requires the referencee_meta_file to assign each element of the batch to the right segmentation
        if os.environ['LEVEL'] == "batch":       
            reference_meta_json_file = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['REFERENCE_META_FILE'])
            print(f"Ref file {reference_meta_json_file}")
            with open(reference_meta_json_file) as json_file:
                ref_meta_json = json.load(json_file)       
            
            for qm_id, qm_data in qm_data_json.items():


                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")    
                    
                json_dict = {}
                json_dict.update({'date': date})
                json_dict.update({'time': time})
                json_dict.update({'study_id': os.environ['STUDY_ID']})       
                #json_dict.update({'study_description': os.environ['STUDY_DESCRIPTION']})                            
                series_uid = os.environ['SERIES_UID']
                if series_uid == '':
                    series_uid = pydicom.uid.generate_uid()                
                json_dict.update({'series_uid': series_uid})
                json_dict.update({'series_number': os.environ['SERIES_NUMBER']})
                json_dict.update({'series_description': os.environ['SERIES_DESCRIPTION']})
                json_dict.update({'patient_first_name': os.environ['PATIENT_FIRST_NAME']})
                json_dict.update({'patient_last_name': os.environ['PATIENT_LAST_NAME']})
                
                instance_uid = os.environ['INSTANCE_UID']
                if instance_uid == '':
                    instance_uid = pydicom.uid.generate_uid()  
                json_dict.update({'instance_uid': instance_uid})
                json_dict.update({'location': os.environ['LOCATION']})

                # Add Study and Patient UID from reference meta file if it was passed
                ref_meta_data = ref_meta_json[qm_id]
                json_dict.update({'study_uid': ref_meta_data['0020000D StudyInstanceUID_keyword']})
                json_dict.update({'ref_series_uid': ref_meta_data['0020000E SeriesInstanceUID_keyword']})
                json_dict.update({'patient_id': ref_meta_data['00100020 PatientID_keyword']})
                
                payload = {
                    'seg_date': ref_meta_data['00080021 SeriesDate_date'],
                    'location': json_dict['location'],
                    'measures': qm_data
                }
                json_dict.update({'data': json.dumps(payload)})
            
                element_output_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], ref_meta_data['0020000E SeriesInstanceUID_keyword'], os.environ['OPERATOR_OUT_DIR'])
                    
                if not os.path.exists(element_output_dir):
                    print(f"Out dir {element_output_dir}")
                    os.makedirs(element_output_dir)
                
                #element_output_dcm = os.path.join(element_output_dir, f'{qm_id}.dcm')
                element_output_xml = os.path.join(element_output_dir, f'data.xml')
                element_output_dcm = os.path.join(element_output_dir, f'json_sr.dcm')
                

                env = Environment(
                    loader=FileSystemLoader('templates'),
                    autoescape=select_autoescape(['html', 'xml'])
                )
                template = env.get_template('template.xml')
                rendered_xml = template.render(json_dict)
                
                with open(element_output_xml, "w") as xml_file:
                    xml_file.write(rendered_xml)
                    
                try:        
                    c = [f"xml2dsr", "data.xml", element_output_dcm]
                    print(f"running command: {c}")
                    resp = subprocess.check_output([
                        f"xml2dsr",
                        element_output_xml, element_output_dcm
                    ], stderr=subprocess.STDOUT)
                    print(element_output_xml)
                    os.remove(element_output_xml)
                except subprocess.CalledProcessError as e:
                    print("DCMTK Error: ")
                    print(e.output)
        
        # level 'element' does not require a reference_meta_file and sets the meta data to the passed parameters
        else:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")    
                
            json_dict = {}
            json_dict.update({'date': date})
            json_dict.update({'time': time})
            json_dict.update({'study_id': os.environ['STUDY_ID']})       
            #json_dict.update({'study_description': os.environ['STUDY_DESCRIPTION']})            
            series_uid = os.environ['SERIES_UID']
            if series_uid == '':
                series_uid = pydicom.uid.generate_uid()                
            json_dict.update({'series_uid': series_uid})
            json_dict.update({'series_number': os.environ['SERIES_NUMBER']})
            json_dict.update({'series_description': os.environ['SERIES_DESCRIPTION']})
            json_dict.update({'patient_first_name': os.environ['PATIENT_FIRST_NAME']})
            json_dict.update({'patient_last_name': os.environ['PATIENT_LAST_NAME']})                        
            instance_uid = os.environ['INSTANCE_UID']
            if instance_uid == '':
                instance_uid = pydicom.uid.generate_uid()     
            json_dict.update({'instance_uid': instance_uid})
            json_dict.update({'location': os.environ['LOCATION']})
            payload = {
                'location': json_dict['location'],
                'measures': qm_data_json
            }
            json_dict.update({'data': json.dumps(payload)})

            # Add Study and Patient UID from passed env vars:
            study_uid = os.environ['STUDY_UID']
            if study_uid == '':
                study_uid = pydicom.uid.generate_uid()
            json_dict.update({'study_uid': study_uid})
            json_dict.update({'patient_id': os.environ['PATIENT_ID']})
            element_output_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['OPERATOR_OUT_DIR'])
            #element_output_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ['BATCH_NAME'], f'{dir_cntr:04d}_json2dcm', os.environ['OPERATOR_OUT_DIR'])
                
            if not os.path.exists(element_output_dir):
                print(f"Out dir {element_output_dir}")
                os.makedirs(element_output_dir)
            
            element_output_xml = os.path.join(element_output_dir, f'data_{dir_cntr:04d}.xml')
            element_output_dcm = os.path.join(element_output_dir, f'json_sr_{dir_cntr:04d}.dcm')

            env = Environment(
                loader=FileSystemLoader('templates'),
                autoescape=select_autoescape(['html', 'xml'])
            )
            template = env.get_template('template.xml')
            rendered_xml = template.render(json_dict)
            
            with open(element_output_xml, "w") as xml_file:
                xml_file.write(rendered_xml)
                
            try:        
                c = [f"xml2dsr", "data_{dir_cntr:04d}.xml", element_output_dcm]
                print(f"running command: {c}")
                resp = subprocess.check_output([
                    f"xml2dsr",
                    element_output_xml, element_output_dcm
                ], stderr=subprocess.STDOUT)
                print(element_output_xml)
                os.remove(element_output_xml)
            except subprocess.CalledProcessError as e:
                print("DCMTK Error: ")
                print(e.output)
                
            dir_cntr += 1    
