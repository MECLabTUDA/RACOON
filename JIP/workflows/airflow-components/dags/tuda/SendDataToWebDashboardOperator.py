# -*- coding: utf-8 -*-

import os
import json
import glob
import requests

from datetime import timedelta, datetime
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR
from airflow.models import Variable

"""
Example Request JSONs:

Daily Data:
{
   "location": "cha",
   "date": "2020-01-21",
   "measureData":[
       {
           "measure":"measure_id",
           "value":"measure_value"
       },
       {
           "measure":"quality_noise",
           "value":"10"
       }
   ],
   "plotData":[
       {
           "name_de": "Plot Name deutsch",
           "name_en": "Plot name englisch",
           "public_visible": true,
           "plot_data": {JSON encoded plot option. Refer to https://echarts.apache.org/examples/en/index.html for examples},
       }
   ]
}

Location:
{
        "location_id": "aug",
        "latitude": "48.36680410",
        "longitude": "10.89869710",
        "name_de": "Augsburg",
        "name_en": "Augsburg",
        "description_de": "Augsburg in Bayern ist eine der ältesten Städte Deutschlands.",
        "description_en": "Augsburg, Bavaria is one of Germany’s oldest cities."
}

Measure:
{
    "measure_id": "count_all",
    "public_visible": true,
    "is_main": true,
    "is_color_default": false,
    "is_size_default": true,
    "is_open_ended": true,
    "lower_bound": 0.0,
    "upper_bound": 0.0,
    "name_de": "Anzahl Gesamt",
    "name_en": "Number Total",
    "description_de": "Die Zahl aller Fälle",
    "description_en": "Number of cases"
}
"""


class SendDataToWebDashboardOperator(KaapanaPythonBaseOperator):   

    def prepareQmJson(self, json_data):
        json_date = json_data.pop('date')

        # Transform the QM Json into the correct format for the HTTP request
        measure_data = []
        for m_name, m_val in json_data.items():
            measure_data.append({'measure': m_name, 'value': m_val})

        req_data = {}
        req_data['location'] = self.location
        req_data['date'] = json_date
        req_data['plotData'] = []
        req_data['measureData'] = measure_data
        req_json = json.dumps(req_data)

        return req_json

    def preparePlotJson(self, plot_data, location, date):
        req_data = {}
        req_data['location'] = location
        req_data['date'] = date
        req_data['measureData'] = []
        req_data['plotData'] = plot_data
        req_json = json.dumps(req_data)

        return req_json

    def createAndSendNNlogPlots(self, nnLog, location, date):
        # Prepare request data
        req_data = {}
        req_data['location'] = location
        req_data['date'] = date
        req_data['measureData'] = []

        # Parse Log
        dataTrainLoss = []
        dataValidationLoss = []
        diceLosses = {}

        for entry in nnLog["epochs"]:
            dataTrainLoss.append([entry["no"], entry["train-loss"]])
            dataValidationLoss.append([entry["no"], entry["validation-loss"]])
            for obj, val in entry["foreground-dice"].items():
                if not obj in diceLosses.keys():
                    diceLosses[obj] = []
                diceLosses[obj].append([entry["no"], val])


        # Create Loss Plot
        plot_options_loss = {}
        #plot_options_loss["title"] = {"text": "Loss Curves", "left": "center"}
        plot_options_loss["tooltip"] = {"trigger": "axis"}
        plot_options_loss["grid"] = {"left":50, "top":10, "right":15, "bottom":35}
        plot_options_loss["xAxis"] = {"type":"value", "name":"Epoch", "nameLocation": 'middle', 'nameGap': 20}
        plot_options_loss["yAxis"] = {"type":"value", "name":"Loss", "nameLocation": 'middle', 'nameGap': 35}
        plot_options_loss["dataZoom"] = {"type":"inside", "orient":"vertical", "filterMode":"none", "minSpan":15}
        series_loss = []
        series_loss.append({"data": dataTrainLoss, "type":"line", "smooth":True, "name":"Training Loss", "symbol": "none"})
        series_loss.append({"data": dataValidationLoss, "type":"line", "smooth":True, "name":"Validation Loss", "symbol": "none"})
        plot_options_loss["series"] = series_loss

        # Add to request dict
        plot_data_loss = {}
        plot_data_loss["public_visible"] = False
        plot_data_loss["name_en"] = "Loss Curves"
        plot_data_loss["name_de"] = "Loss Kurven"
        plot_data_loss["plot_data"] = plot_options_loss
        req_data['plotData'].append(plot_data_loss)


        # Create Dice Plot:
        plot_options_dice = {}
        #plot_options_dice["title"] = {"text": "Dice Scores", "left": "center"}
        plot_options_dice["tooltip"] = {"trigger": "axis"}
        plot_options_dice["grid"] = {"left":50, "top":10, "right":15, "bottom":35}
        plot_options_dice["xAxis"] = {"type":"value", "name":"Epoch", "nameLocation": 'middle', 'nameGap': 20}
        plot_options_dice["yAxis"] = {"type":"value", "name":"Dice Score", "nameLocation": 'middle', 'nameGap': 35}
        plot_options_dice["dataZoom"] = {"type":"inside", "orient":"vertical", "filterMode":"none", "minSpan":15}
        series_dice = []
        for name, data in diceLosses.items():
            series_dice.append({"data": data, "type":"line", "smooth":True, "name":name, "symbol": "none"})
        plot_options_dice["series"] = series_dice

        # Add to request dict
        plot_data_dice = {}
        plot_data_dice["public_visible"] = False
        plot_data_dice["name_en"] = "Dice Scores"
        plot_data_dice["name_de"] = "Dice Scores"
        plot_data_dice["plot_data"] = plot_options_dice
        req_data['plotData'].append(plot_data_dice)

        # Send Request       
        req_json = json.dumps(req_data)
        self.sendRequest(req_json, api_endpoint='data', action="createOrUpdate")


    def sendRequest(self, req_json, api_endpoint="data", action="createOrUpdate"):
        url = f"{self.dashboard_web_host}:{self.dashboard_web_port}/api/{api_endpoint}/"
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {self.apiKey}'}

        print(f"URL: {url}")
        print(f"Headers: {headers}")
        print(f"Action: {action}")
        print(f"Data: {req_json}")

        if action == "update":
            r_patch = requests.patch(url, headers=headers, data=req_json)
            print("Sending PATCH Request")
            print(f" Response: {r_patch.status_code} - {str(r_patch.json())}")

        elif action == "create":
            r_post = requests.post(url, headers=headers, data=req_json)
            print("Sending POST Request")
            print(f"Response: {r_post.status_code} - {str(r_post.json())}")

        elif action == "createOrUpdate":
            r_post = requests.post(url, headers=headers, data=req_json)
            print("Sending POST Request")
            print(f"Response: {r_post.status_code} - {str(r_post.json())}")
            if r_post.status_code == 400:
                if "must make a unique set" in str(r_post.json()):
                    print(" Data entry (location - date) already exists - attempting to update (PATCH) existing instance:")
                    r_patch = requests.patch(url, headers=headers, data=req_json)
                    print("Sending PATCH Request")
                    print(f" Response: {r_patch.status_code} - {str(r_patch.json())}")

    def processDir(self, jsonFileDir):
        jsonFiles = glob.glob(os.path.join(jsonFileDir, "*.json"))
        for jsonFile in jsonFiles:
            print(f"Processing {jsonFile}")

            with open(jsonFile) as data_file:
                json_data = json.load(data_file)

            # Send data from a JSON file created by DcmSr2Json.py
            if self.command == 'sendQualityMeasures':
                json_req = self.prepareQmJson(json_data)
                self.sendRequest(json_req, api_endpoint='data', action="createOrUpdate")

            # Create a new location on the dashboard (Location is JSON encoded in variable 'data')
            elif self.command == 'createLocation':
                self.sendRequest(json_data, api_endpoint='locations', action="ceate")

            # Create a new measure on the dashboard (Measure is JSON encoded in variable 'data')
            elif self.command == 'createMeasure':
                self.sendRequest(json_data, api_endpoint='measures', action="create")

            # Send a new plot to a specific location-date POI (Plot is JSON encoded in variable 'data')
            elif self.command == 'sendPlots':
                json_req = self.preparePlotJson(json_data, self.location, self.date)
                self.sendRequest(json_req, api_endpoint='data', action="createOrUpdate")

            # Send data from a NN run (log has to be included in variable 'data')
            elif self.command == 'sendNNtrainingLog':
                self.createAndSendNNlogPlots(json_data, self.location, self.date)

            else:
                raise ValueError('command must be in [createLocation", "createMeasure", "sendPlots", "sendNNtrainingLog", "sendQualityMeasures"]')


    def start(self, ds, **kwargs):
        # If json dir is passed process files from there
        if self.json_dir != '':
            self.processDir(self.json_dir)
        # Otherwise process files from default input dir batch folders
        else:
            dag_run_id = kwargs['dag_run'].run_id
            batch_folders = [f for f in glob.glob(os.path.join(WORKFLOW_DIR, dag_run_id, BATCH_NAME, '*'))]

            for batch_element_dir in batch_folders:            
                element_input_dir = os.path.join(batch_element_dir, self.operator_in_dir)
                self.processDir(element_input_dir)


    def __init__(self,
                 dag,
                 command, # must be in ["createLocation", "createMeasure", "sendPlots", "sendNNtrainingLog", "sendQualityMeasures"]
                 location='', # optional, if not set the node_uid will be used
                 date='today',
                 dashboard_web_host='DASHBOARD URL',
                 dashboard_web_port='DASHBOARD PORT',
                 dashboard_api_token='DASHBOARD TOKEN',
                 json_dir='', # Optional - Load Json from this dir (instead of input_operator.operator_out_dir)
                 *args,
                 **kwargs):
        
        self.json_dir = json_dir

        self.command = command

        self.dashboard_web_host = dashboard_web_host
        self.dashboard_web_port = dashboard_web_port
        self.apiKey = dashboard_api_token

        self.location = location
        if location == '':
            # extract location ID from JIP
            self.location = Variable.get(key="node_uid", default_var="N/A")
        
        self.date = date
        if date == '' or date.lower() == 'today':
            now = datetime.now()
            self.date = now.strftime('%Y-%m-%d')      

        super().__init__(
            dag,
            name="sendDataToWebDashbord",
            python_callable=self.start,
            task_concurrency=10,
            execution_timeout=timedelta(minutes=15),
            *args, **kwargs
        )
