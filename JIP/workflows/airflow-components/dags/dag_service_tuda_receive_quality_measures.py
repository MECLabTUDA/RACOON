from kaapana.operators.LocalGetInputDataOperator import LocalGetInputDataOperator
from tuda.DcmSr2JsonOperator import DcmSr2JsonOperator
from tuda.IdentifyJsonDsrOperator import IdentifyJsonDsrOperator
from tuda.SendDataToWebDashboardOperator import SendDataToWebDashboardOperator
from kaapana.operators.LocalWorkflowCleanerOperator import LocalWorkflowCleanerOperator

from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import DAG
import os


log = LoggingMixin().log

# ui_forms = {
#     "workflow_form": {
#         "type": "object",
#         "properties": {
#             "single_execution": {
#                 "title": "single execution",
#                 "description": "Should each series be processed separately?",
#                 "type": "boolean",
#                 "default": False,
#                 "readOnly": False,
#             }
#         }
#     }
# }

args = {
    # 'ui_forms': ui_forms,
    'ui_visible': False,
    'owner': 'kaapana',
    'start_date': days_ago(0),
    'retries': 0,
    'retry_delay': timedelta(seconds=30)
}

dag = DAG(
    dag_id='service-tuda-receive-quality-measures',
    default_args=args,
    schedule_interval=None
    )


get_input = LocalGetInputDataOperator(dag=dag)
identify_json_dsr = IdentifyJsonDsrOperator(dag=dag, input_operator=get_input, series_description_filter="Aggregated")
dcm_to_json = DcmSr2JsonOperator(dag=dag, input_operator=identify_json_dsr)
send_to_dashboard = SendDataToWebDashboardOperator(dag=dag, 
                                                   input_operator=dcm_to_json, 
                                                   command="sendQualityMeasures", 
                                                   dashboard_root_url="http://dashboard-service.base.svc:5001", # racoon-dashboard not needed, since we are in the backend
                                                   dashboard_api_token="cef4d40440c2411a22fe5635c54ee8501b4bfe53")
clean = LocalWorkflowCleanerOperator(dag=dag)

get_input >> identify_json_dsr >> dcm_to_json >> send_to_dashboard >> clean