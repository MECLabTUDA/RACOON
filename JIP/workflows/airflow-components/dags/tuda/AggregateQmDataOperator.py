# -*- coding: utf-8 -*-

import os
import json
import glob
import requests
from datetime import datetime

from datetime import timedelta, datetime
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR



class AggregateQmDataOperator(KaapanaPythonBaseOperator):

    def start(self, ds, **kwargs):
        if self.aggregation_strategy == "all":
            self.aggregate_all(**kwargs)
        elif self.aggregation_strategy == "seg_date":
            self.aggregate_by_seg_date(**kwargs)
        else:
            print("'aggregation_strategy' has to be in ['all', 'seg_date']")
            exit(1)

    def aggregate_by_seg_date(self, **kwargs):
        run_id = kwargs['run_id']

        batch_folders = [f for f in glob.glob(os.path.join(WORKFLOW_DIR, run_id, BATCH_NAME, '*'))]
        aggregation_results = {}
        aggregation_counter = {}
        global_counter = {}
        dates = []

        for batch_element_dir in batch_folders:
            
            element_input_dir = os.path.join(batch_element_dir, self.operator_in_dir)

            for jsonFile in glob.glob(os.path.join(element_input_dir, "*.json")):
                print(f"Processing {jsonFile}")

                with open(jsonFile) as data_file:
                    json_data = json.load(data_file)
                
                    if "seg_date" in json_data:
                        date = json_data.pop("seg_date")
                    else :
                        print("Error: For 'Aggregating by Segmentation date' the attribute 'seg_date' is required inside the qm json.")
                        exit(1)

                    if date not in dates:
                        dates.append(date)

                    if date not in global_counter:
                        global_counter[date] = 0

                    global_counter[date] += 1

                    for m_name, m_val in json_data.items():
                        if date not in aggregation_results:
                            aggregation_results[date] = {}
                            aggregation_counter[date] = {}
                        if m_name not in aggregation_results[date]:
                            aggregation_results[date][m_name] = 0
                            aggregation_counter[date][m_name] = 0
                        aggregation_results[date][m_name] += m_val
                        aggregation_counter[date][m_name] += 1


        models_dir = os.path.join("/models/tuda/aggregations/")
        if not os.path.exists(models_dir):
            print(f"Aggregation Database does not yet exist - Creating JSON Database dir: {models_dir}")
            os.makedirs(models_dir)

        # load the aggregation database JSON. This file contains the aggregations from all dates.
        # It compared to the newly calculated aggregations to make sure that only date entries where aggreations changed are stored and sent to central
        aggregation_database_file = os.path.join(models_dir, "aggregation_db.json") 
        if os.path.exists(aggregation_database_file):
            with open(aggregation_database_file) as json_file:
                aggregation_database = json.load(json_file)   
        else:
            aggregation_database = {} 

        # finalize the aggregations
        for date in list(set(dates)):
            for m_name, m_val in aggregation_results[date].items():
                aggregation_results[date][m_name] = m_val / aggregation_counter[date][m_name]

            aggregation_results[date]['count_total'] = global_counter[date]
            aggregation_results[date]['date'] = date

            # create output dir
            element_output_dir = os.path.join(WORKFLOW_DIR, run_id, self.operator_out_dir)
            if not os.path.exists(element_output_dir):
                os.makedirs(element_output_dir)
            element_output_file = os.path.join(element_output_dir, f'aggregated_metrics_{date}.json')

            if date in aggregation_database:
                # only store new aggregation DICOM, if it exists not in the aggregation database or differs from the database entry (which means it is either new or was altered)
                if not aggregation_database[date] == aggregation_results[date]:
                    # update database
                    print(f"Updating database for date: {date}")
                    aggregation_database[date] = aggregation_results[date]
                    # save aggregation json
                    with open(element_output_file, "w") as fp:
                        json.dump(aggregation_results[date], fp)
            
            else:
                # create database entry
                print(f"New Aggregation Date added to database: {date}")
                aggregation_database[date] = aggregation_results[date]
                # save aggregation json
                with open(element_output_file, "w") as fp:
                    json.dump(aggregation_results[date], fp)

        # Save updated aggregation database
        with open(aggregation_database_file, "w") as fp:
            json.dump(aggregation_database, fp)
            
    def aggregate_all(self, **kwargs):
        run_id = kwargs['run_id']

        batch_folders = [f for f in glob.glob(os.path.join(WORKFLOW_DIR, run_id, BATCH_NAME, '*'))]
        aggregation_results = {}
        aggregation_counter = {}
        global_counter = 0


        for batch_element_dir in batch_folders:
            
            element_input_dir = os.path.join(batch_element_dir, self.operator_in_dir)

            for jsonFile in glob.glob(os.path.join(element_input_dir, "*.json")):
                print(f"Processing {jsonFile}")
                global_counter += 1

                with open(jsonFile) as data_file:
                    json_data = json.load(data_file)

                    # remove seg_date from json data since it is only required for the aggreagation by date
                    if "seg_date" in json_data:
                        json_data.pop("seg_date")

                    for m_name, m_val in json_data.items():
                        if m_name not in aggregation_results:
                            aggregation_results[m_name] = 0
                            aggregation_counter[m_name] = 0
                        aggregation_results[m_name] += m_val
                        aggregation_counter[m_name] += 1


        for m_name, m_val in aggregation_results.items():
            aggregation_results[m_name] = m_val / aggregation_counter[m_name]

        aggregation_results['count_total'] = global_counter
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')   
        aggregation_results['date'] = date

        element_output_dir = os.path.join(WORKFLOW_DIR, run_id, self.operator_out_dir)
        if not os.path.exists(element_output_dir):
            os.makedirs(element_output_dir)

        element_output_file = os.path.join(element_output_dir, 'aggregated_metrics.json')

        with open(element_output_file, "w") as fp:
            json.dump(aggregation_results, fp)

    def __init__(self,
                 dag,
                 aggregation_strategy,
                 *args,
                 **kwargs):

        self.aggregation_strategy = aggregation_strategy

        super().__init__(
            dag,
            name="aggregateQmData",
            python_callable=self.start,
            task_concurrency=10,
            execution_timeout=timedelta(minutes=15),
            *args, **kwargs
        )
