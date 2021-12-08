
from datetime import datetime
from datetime import timedelta
from kaapana.operators.KaapanaBaseOperator import KaapanaBaseOperator, default_registry, default_project
from airflow.models import Variable

class Json2DcmSrOperator(KaapanaBaseOperator):

    def __init__(self,
                 dag,
                 level, # 'element' of 'batch'
                 reference_meta_file='', # required for level 'batch'
                 patient_id="123", 
                 patient_first_name="Max",
                 patient_last_name="Muster",
                 study_id="json2dcm",
                 study_uid='', # will be randomly generated, if not set
                 series_uid='', # will be randomly generated, if not set
                 series_number="1",
                 series_description="", 
                 instance_uid='', # will be randomly generated, if not set
                 env_vars=None,
                 execution_timeout=timedelta(minutes=5),
                 *args,
                 **kwargs):

        if level not in ['element', 'batch']:
            raise NameError('level must be either "element" or "batch". \
                If element, an operator folder next to the batch folder will be created that contains all .dcm files. \
                If batch, *.dcm will be stored inside the corresponding batch folders. This level also requires a "reference_meta_file" to be set in order to assign the DCM-SR to the correct studies'
                            )

        # extract location ID from JIP
        node_uid = Variable.get(key="racoon_node_id", default_var="N/A")

        if env_vars is None:
            env_vars = {}

        envs = {
            "DCMTK_COMMAND": "json2dcm",
            "LEVEL": level,
            "REFERENCE_META_FILE": reference_meta_file, # created by PrepareInputDataOperator. Required for level == "batch" to assign json data to corresponding img-seg pair            
            "STUDY_ID": study_id,
            "STUDY_UID": study_uid,
            "SERIES_UID": series_uid,
            "SERIES_NUMBER": series_number,
            "SERIES_DESCRIPTION": f"JSON Embedding - {series_description}",
            "PATIENT_ID": patient_id,
            "PATIENT_FIRST_NAME": patient_first_name,
            "PATIENT_LAST_NAME": patient_last_name,
            "INSTANCE_UID": instance_uid,
            "LOCATION": node_uid
        }
        env_vars.update(envs)

        if level == "batch" and reference_meta_file == "":
            print("---------------------------------------------------")
            print("Error: For 'batch' level a reference meta file must be passed that was created by the PrepareInputDataOperator in order to assign the json files to their corresponding img-seg pairs")
            print("---------------------------------------------------")
            exit(1)

        super().__init__(
            dag=dag,
            image="{}{}/tuda-dcmtools:0.1.0".format(default_registry, default_project),  # Image from docker container /processing-container/jsonDcmSr_tools
            name="json2dcm",
            env_vars=env_vars,
            image_pull_secrets=["registry-secret"],
            execution_timeout=execution_timeout,
            ram_mem_mb=1000,
            *args,
            **kwargs
            )
