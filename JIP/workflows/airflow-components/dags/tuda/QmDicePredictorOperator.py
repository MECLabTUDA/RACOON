
from datetime import timedelta
from kaapana.operators.KaapanaBaseOperator import KaapanaBaseOperator, default_registry, default_project
from kaapana.kubetools.volume_mount import VolumeMount
from kaapana.kubetools.volume import Volume
from kaapana.kubetools.resources import Resources as PodResources
import os 

class QmDicePredictorOperator(KaapanaBaseOperator):

    def __init__(self,
                 dag,
                 env_vars=None,
                 execution_timeout=timedelta(minutes=10),
                 *args,
                 **kwargs):

        # Environmental Vars initialized in container
        if env_vars is None:
            env_vars = {}

        envs = {
            "PERSISTENT_DIR": "/models/tuda/",
        }
        env_vars.update(envs)

        data_dir = os.getenv('DATADIR', "")
        models_dir = os.path.join(os.path.dirname(data_dir), "models")

        training_operator = True
        pod_resources = PodResources(request_memory=None, request_cpu=None, limit_memory=None, limit_cpu=None, limit_gpu=None)
        gpu_mem_mb = 11000

        super().__init__(
            dag=dag,
            image="{}{}/tuda_qm_dice:0.1.0".format(default_registry, default_project),  # Image from docker container /processing-container/qm_dicePredictor
            name=f'qm-dice-predictor',
            env_vars=env_vars,
            image_pull_secrets=["registry-secret"],
            execution_timeout=execution_timeout,
            ram_mem_mb=None,
            ram_mem_mb_lmt=None,
            pod_resources=pod_resources,
            training_operator=training_operator,
            gpu_mem_mb=gpu_mem_mb,
            *args,
            **kwargs
        )