from tuda.GetAllQmDsrOperator import GetAllQmDsrOperator
from tuda.Json2DcmSrOperator import Json2DcmSrOperator
from tuda.DcmSr2JsonOperator import DcmSr2JsonOperator
from tuda.AggregateQmDataOperator import AggregateQmDataOperator
from kaapana.operators.DcmSendOperator import DcmSendOperator
from kaapana.operators.LocalWorkflowCleanerOperator import LocalWorkflowCleanerOperator

from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import DAG
import os


log = LoggingMixin().log

args = {
    'ui_visible': False,
    'owner': 'kaapana',
    'start_date': days_ago(0),
    'retries': 0,
    'retry_delay': timedelta(seconds=30)
}

dag = DAG(
    dag_id='tuda_aggregate-quality-measures',
    default_args=args,
    schedule_interval='@daily',
    max_active_runs=1
    )


get_all_qm_dsr = GetAllQmDsrOperator(dag=dag)
dcm_to_json = DcmSr2JsonOperator(dag=dag, input_operator=get_all_qm_dsr)
aggregate = AggregateQmDataOperator(dag=dag, input_operator=dcm_to_json, aggregation_strategy='seg_date')
json_to_dcm = Json2DcmSrOperator(dag=dag, 
    input_operator=aggregate, 
    level='element', 
    series_description="Aggregated QM",
    study_uid="1.2.826.0.1.3680043.8.498.12778348761350828823404074873274303479", 
    patient_id="20202020", 
    patient_first_name="QM", 
    patient_last_name="Statistics")

dcm_send = DcmSendOperator(dag=dag, input_operator=json_to_dcm, level='batch')
clean = LocalWorkflowCleanerOperator(dag=dag)

get_all_qm_dsr >> dcm_to_json >> aggregate >> json_to_dcm >> dcm_send >> clean