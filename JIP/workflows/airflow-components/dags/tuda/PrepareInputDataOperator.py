# -*- coding: utf-8 -*-

import os
import json
import glob
import pydicom

from datetime import timedelta
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR

from kaapana.operators.HelperDcmWeb import HelperDcmWeb
from kaapana.operators.HelperElasticsearch import HelperElasticsearch

from multiprocessing.pool import ThreadPool
from dicomweb_client.api import DICOMwebClient

from pathlib import Path


class PrepareInputDataOperator(KaapanaPythonBaseOperator):

    def get_corresponding_img(self, incoming_dcm, target_img_dir): 
        client = DICOMwebClient(url=self.pacs_dcmweb, qido_url_prefix="rs", wado_url_prefix="rs", stow_url_prefix="rs")
        if (0x0008, 0x1115) in incoming_dcm:
            for ref_series in incoming_dcm[0x0008, 0x1115]:
                if (0x0020, 0x000E) not in ref_series:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Could not extract SeriesUID from referenced DICOM series.")
                    print("Abort Reference Image Download.")
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    exit(1)
                reference_series_uid = ref_series[0x0020, 0x000E].value
                pacs_series = client.search_for_series(search_filters={'0020000E': reference_series_uid})
                print(f"Found series: {len(pacs_series)} for reference_series_uid: {reference_series_uid}")
                if len(pacs_series) != 1:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print(f"Could not find referenced SeriesUID in the PACS: {reference_series_uid} !")
                    print("Abort Reference Image Download.")
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    exit(1)
                HelperDcmWeb.downloadSeries(seriesUID=reference_series_uid, target_dir=target_img_dir)
                print(f"Downloading Reference Image")
                print(f"   seriesUID: {reference_series_uid} ")
                print(f"   targetDir: {target_img_dir}")


    def get_seg_data_from_input_op(self, **kwargs):
        meta_data = {}

        dag_run_id = kwargs['dag_run'].run_id

        batch_folders = [f for f in glob.glob(os.path.join(WORKFLOW_DIR, dag_run_id, BATCH_NAME, '*'))]
        for batch_element_dir in batch_folders:
            batch_element_out_dir = os.path.join(batch_element_dir, self.operator_out_dir)
            dcm_files = sorted(glob.glob(os.path.join(batch_element_dir, self.operator_in_dir, "*.dcm*"), recursive=True))

            dcm_files = sorted(glob.glob(os.path.join(batch_element_dir, self.operator_in_dir, "*.dcm*"), recursive=True))
            if len(dcm_files) > 0:
                incoming_dcm = pydicom.dcmread(dcm_files[0])

                modality = incoming_dcm.Modality
                series_uid = incoming_dcm.SeriesInstanceUID
                sop_inst_uid = incoming_dcm.SOPInstanceUID

                print(f"Modality: {series_uid}")
                
                if modality.lower() == "seg":
                    seg_target_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir, f'{self.dir_cntr:04d}', "seg")
                    img_target_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir, f'{self.dir_cntr:04d}', "img")

                    # Download SEG
                    print(f"Preparing Segmentation")
                    print(f"   seriesUID: {series_uid} ")
                    print(f"   targetDir: {seg_target_dir}") 
                    metadata_json_path = os.path.join(batch_element_dir, self.src_meta_operator.operator_out_dir, f"{series_uid}.json")
                    print(f"Metadata JSON: {metadata_json_path}")
                    with open(metadata_json_path) as fp:
                        meta_data[f'{self.dir_cntr:04d}'] = json.load(fp)
                    # tuda_calc_qualty_measures might be triggered after images are in PACS but before metadata are in meta-index
                    #meta_data[f'{self.dir_cntr:04d}'] = HelperElasticsearch.get_series_metadata(series_uid)
                    if not os.path.exists(seg_target_dir):
                        os.makedirs(seg_target_dir)
                    pydicom.dcmwrite(os.path.join(seg_target_dir, f'{sop_inst_uid}.dcm'), incoming_dcm)
                    self.dir_cntr += 1 

                    # Download Corresponding IMG
                    self.get_corresponding_img(incoming_dcm, img_target_dir)
                else:
                    print(f"Skipping series {series_uid} - Reason: Wrong modality, got {modality}, expected SEG")


        if meta_data:
            element_output_json_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir)
            if not os.path.exists(element_output_json_dir):
                os.makedirs(element_output_json_dir)
            element_output_json = os.path.join(element_output_json_dir, "reference_meta.json")
            with open(element_output_json, 'w') as outfile:
                json.dump(meta_data, outfile)   


    def get_seg_data_from_conf(self, **kwargs):
        meta_data = {}

        self.conf = kwargs['dag_run'].conf
        dag_run_id = kwargs['dag_run'].run_id

        if self.conf == None or not "inputs" in self.conf:
            print("No config or inputs in config found!")
            print("Skipping...")
            return

        inputs = self.conf["inputs"]

        if not isinstance(inputs, list):
            inputs = [inputs]

        for input in inputs:
            if "elastic-query" in input:
                elastic_query = input["elastic-query"]
                if "query" not in elastic_query:
                    print("'query' not found in 'elastic-query': {}".format(input))
                    print("abort...")
                    exit(1)
                if "index" not in elastic_query:
                    print("'index' not found in 'elastic-query': {}".format(input))
                    print("abort...")
                    exit(1)

                query = elastic_query["query"]
                index = elastic_query["index"]

                cohort = HelperElasticsearch.get_query_cohort(elastic_index=index, elastic_query=query)

                for series in cohort:

                    series = series["_source"]

                    study_uid = series[HelperElasticsearch.study_uid_tag]
                    series_uid = series[HelperElasticsearch.series_uid_tag]
                    modality = series[HelperElasticsearch.modality_tag]
               
                    if modality.lower() == "seg":
                        seg_target_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir, f'{self.dir_cntr:04d}', "seg")
                        img_target_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir, f'{self.dir_cntr:04d}', "img")
                        
                        # Download SEG
                        print(f"Downloading Segmentation")
                        print(f"   seriesUID: {series_uid} ")
                        print(f"   targetDir: {seg_target_dir}") 
                        meta_data[f'{self.dir_cntr:04d}'] = HelperElasticsearch.get_series_metadata(series_uid)
                        HelperDcmWeb.downloadSeries(seriesUID=series_uid, target_dir=seg_target_dir)
                        self.dir_cntr += 1

                        # Download Corresponding IMG
                        seg_dcm = pydicom.dcmread(glob.glob(seg_target_dir + "/*.dcm")[0])
                        self.get_corresponding_img(seg_dcm, img_target_dir)
                    else:
                        print(f"Skipping series {series_uid} - Reason: Wrong modality, got {modality}, expected SEG")
                
                            

            elif "dcm-uid" in input:
                dcm_uid = input["dcm-uid"]

                if "study-uid" not in dcm_uid:
                    print("'study-uid' not found in 'dcm-uid': {}".format(input))
                    print("abort...")
                    exit(1)
                if "series-uid" not in dcm_uid:
                    print("'series-uid' not found in 'dcm-uid': {}".format(input))
                    print("abort...")
                    exit(1)

                study_uid = dcm_uid["study-uid"]
                series_uid = dcm_uid["series-uid"]

                if "modality" in dcm_uid:
                    modality = dcm_uid["modality"]                    
                    if modality.lower() == "seg":
                        seg_target_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir, f'{self.dir_cntr:04d}', "seg")
                        img_target_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir, f'{self.dir_cntr:04d}', "img")

                        # Download SEG
                        print(f"Downloading Segmentation")
                        print(f"   seriesUID: {series_uid} ")
                        print(f"   targetDir: {seg_target_dir}")  
                        meta_data[f'{self.dir_cntr:04d}'] = HelperElasticsearch.get_series_metadata(series_uid)
                        HelperDcmWeb.downloadSeries(seriesUID=series_uid, target_dir=seg_target_dir)
                        self.dir_cntr += 1
                        
                        # Download Corresponding IMG  
                        seg_dcm = pydicom.dcmread(glob.glob(seg_target_dir + "/*.dcm")[0])
                        self.get_corresponding_img(seg_dcm, img_target_dir)
                    else:
                        print(f"Skipping series {series_uid} - Reason: Wrong modality, got {modality}, expected SEG")
                     

            else:
                print("Error with dag-config!")
                print("Unknown input: {}".format(input))
                print("Supported 'dcm-uid' and 'elastic-query' ")
                print("Dag-conf: {}".format(self.conf))
                exit(1)
        
        if meta_data:
            element_output_json_dir = os.path.join(WORKFLOW_DIR, dag_run_id, self.operator_out_dir)
            if not os.path.exists(element_output_json_dir):
                os.makedirs(element_output_json_dir)
            element_output_json = os.path.join(element_output_json_dir, "reference_meta.json")
            with open(element_output_json, 'w') as outfile:
                json.dump(meta_data, outfile)


    def start(self, ds, **kwargs):
        if hasattr(self, 'operator_in_dir'):
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Preparing data from input op: {self.operator_in_dir} !")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.get_seg_data_from_input_op(**kwargs)
        else:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Preparing data from config !")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.get_seg_data_from_conf(**kwargs)

    def __init__(self,
                 dag,
                 pacs_dcmweb_host='http://dcm4chee-service.store.svc',
                 pacs_dcmweb_port='8080',
                 aetitle="KAAPANA",
                 src_meta_operator="",
                 *args,
                 **kwargs):

        self.pacs_dcmweb = pacs_dcmweb_host+":"+pacs_dcmweb_port + "/dcm4chee-arc/aets/"+aetitle.upper()

        self.dir_cntr = 1
        self.download_series_seg_list = []
        self.download_series_img_list = []
        self.src_meta_operator = src_meta_operator

        super().__init__(
            dag,
            name="prepare-input-data",
            python_callable=self.start,
            task_concurrency=10,
            execution_timeout=timedelta(minutes=15),
            *args, **kwargs
        )