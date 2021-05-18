
from datetime import datetime
from datetime import timedelta
from kaapana.operators.KaapanaBaseOperator import KaapanaBaseOperator, default_registry, default_project

class DcmSr2JsonOperator(KaapanaBaseOperator):

    def __init__(self,
                 dag,
                 output_json_filename="metrics.json",
                 env_vars=None,
                 execution_timeout=timedelta(minutes=5),
                 *args,
                 **kwargs):

        if env_vars is None:
            env_vars = {}

        envs = {
            "DCMTK_COMMAND": "dcm2json",
            "OUTPUT_JSON_FILENAME": output_json_filename
        }
        env_vars.update(envs)

        my_registry = "docker.io"
        my_project = "/tudracoon"

        super().__init__(
            dag=dag,
            image="{}{}/publictest:dcmTools_88".format(my_registry, my_project), # Image from docker container /processing-container/jsonDcmSr_tools
            name="dcm2json",
            env_vars=env_vars,
            image_pull_secrets=["registry-secret"],
            execution_timeout=execution_timeout,
            ram_mem_mb=1000,
            *args,
            **kwargs
            )
