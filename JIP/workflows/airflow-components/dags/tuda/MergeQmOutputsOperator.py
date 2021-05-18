# -*- coding: utf-8 -*-

import os
import json
import glob
import requests
import pydicom

from datetime import timedelta, datetime
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR

class MergeQmOutputsOperator(KaapanaPythonBaseOperator):

    def start(self, ds, **kwargs):
        run_id = kwargs['run_id']

        qmJsons = []
        # Search for JSON files in previous operator output dir
        for op in self.qm_operators:
            print(f"Search for QM JSON in: {os.path.join(WORKFLOW_DIR, run_id, op.operator_out_dir)}")
            print(f"Found: {glob.glob(os.path.join(WORKFLOW_DIR, run_id, op.operator_out_dir, '*.json'))}")
            qmJsons += glob.glob(os.path.join(WORKFLOW_DIR, run_id, op.operator_out_dir, '*.json'))

        result = {}
        # merge all found JSON files
        for fn in qmJsons:
            print(f"Processing: {fn}")
            with open(fn, 'r') as infile:
                data = json.load(infile)
                for key in data:
                    if key in result:
                        result[key].update(data[key])
                    else:
                        result[key] = data[key]

        element_output_dir = os.path.join(WORKFLOW_DIR, run_id, self.operator_out_dir)
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        element_output_file = os.path.join(element_output_dir, 'metrics.json')

        print(result)
        with open(element_output_file, "w") as fp:
            json.dump(result, fp)

        

    def __init__(self,
                 dag,
                 qm_operators,
                 *args,
                 **kwargs):    

        self.qm_operators = qm_operators

        super().__init__(
            dag,
            name="mergeQmOutputs",
            python_callable=self.start,
            task_concurrency=10,
            execution_timeout=timedelta(minutes=15),
            *args, **kwargs
        )
