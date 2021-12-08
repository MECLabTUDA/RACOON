from kaapana.operators.LocalGetInputDataOperator import LocalGetInputDataOperator
from tuda.PrepareInputDataOperator import PrepareInputDataOperator
from tuda.Dcm2ItkOperator import Dcm2ItkOperator
from tuda.QmArtifactsOperator import QmArtifactsOperator
from tuda.QmDicePredictorOperator import QmDicePredictorOperator
from tuda.Json2DcmSrOperator import Json2DcmSrOperator
from tuda.MergeQmOutputsOperator import MergeQmOutputsOperator
from kaapana.operators.DcmSendOperator import DcmSendOperator
from kaapana.operators.LocalWorkflowCleanerOperator import LocalWorkflowCleanerOperator

from airflow.utils.log.logging_mixin import LoggingMixin
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.models import DAG
import os


log = LoggingMixin().log

ui_forms = {
    "workflow_form": {
        "type": "object",
        "properties": {
            "single_execution": {
                "title": "single execution",
                "description": "Should each series be processed separately?",
                "type": "boolean",
                "default": False,
                "readOnly": False,
            }
        }
    }
}

args = {
    'ui_forms': ui_forms,
    'ui_visible': True,
    'owner': 'kaapana',
    'start_date': days_ago(0),
    'retries': 0, 
    'retry_delay': timedelta(seconds=30)
}
 
dag = DAG(
    dag_id='tuda-calc-quality-measures',
    default_args=args,
    schedule_interval=None
    )


get_input = LocalGetInputDataOperator(dag=dag)

# The following block does not fit into the regular JIP directory structure
# =========================================================================
prepare_data = PrepareInputDataOperator(dag=dag, input_operator=get_input)
transform_data = Dcm2ItkOperator(dag=dag, input_operator=prepare_data, output_format='nii.gz')
# List of QM Operators
qm_artifacts = QmArtifactsOperator(dag=dag, input_operator=transform_data)
qm_dice_pred = QmDicePredictorOperator(dag=dag, input_operator=transform_data)
# Merge results of QM Operators
qm_result_merger = MergeQmOutputsOperator(dag=dag, qm_operators=[qm_artifacts, qm_dice_pred])
# =========================================================================

json_to_dcm = Json2DcmSrOperator(dag=dag, input_operator=qm_result_merger, level="batch", series_description="Single QM", reference_meta_file=os.path.join(prepare_data.operator_out_dir, "reference_meta.json"))
dcm_send = DcmSendOperator(dag=dag, input_operator=json_to_dcm, level='element')
clean = LocalWorkflowCleanerOperator(dag=dag, clean_workflow_dir=True)


get_input >> prepare_data >> transform_data >> [qm_artifacts, qm_dice_pred] >> qm_result_merger >> json_to_dcm >> dcm_send >> clean