# -*- coding: utf-8 -*-

import os
import json
import glob
import requests
import pydicom

from datetime import timedelta, datetime
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR

class IdentifyJsonDsrOperator(KaapanaPythonBaseOperator):

    def start(self, ds, **kwargs):
        run_id = kwargs['run_id']

        batch_folders = [f for f in glob.glob(os.path.join(WORKFLOW_DIR, run_id, BATCH_NAME, '*'))]

        for batch_element_dir in batch_folders:
            
            element_input_dir = os.path.join(batch_element_dir, self.operator_in_dir)

            dcm_files = glob.glob(os.path.join(element_input_dir, "*.dcm"))
            if len(dcm_files) > 0:
                incoming_dcm = pydicom.dcmread(dcm_files[0])

                try:
                    modality = incoming_dcm.Modality
                    series_description = incoming_dcm.SeriesDescription
                    sop_inst_uid = incoming_dcm.SOPInstanceUID

                    if modality.lower() == "sr" and "JSON Embedding" in series_description and self.series_description_filter in series_description:

                        print(f"Processing {dcm_files[0]}")

                        element_output_dir = os.path.join(batch_element_dir, self.operator_out_dir)
                        if not os.path.exists(element_output_dir):
                            os.makedirs(element_output_dir)

                        element_output_file = os.path.join(element_output_dir, f'{sop_inst_uid}.dcm')
                        pydicom.dcmwrite(element_output_file, incoming_dcm)
                
                except AttributeError:
                    # Case if series has no attribute Modality, SeriesDescription or SOPInstanceUID -> skip series
                    continue

    def __init__(self,
                 dag,
                 series_description_filter='',
                 *args,
                 **kwargs):    

        self.series_description_filter = series_description_filter

        super().__init__(
            dag,
            name="identifyJsonDsr",
            python_callable=self.start,
            task_concurrency=10,
            execution_timeout=timedelta(minutes=15),
            *args, **kwargs
        )
