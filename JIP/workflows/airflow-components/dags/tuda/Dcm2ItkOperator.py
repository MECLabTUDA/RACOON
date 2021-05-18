
from datetime import datetime
from datetime import timedelta
from kaapana.operators.KaapanaBaseOperator import KaapanaBaseOperator, default_registry, default_project

class Dcm2ItkOperator(KaapanaBaseOperator):

    def __init__(self,
                 dag,
                 output_format='nii',
                 env_vars=None,
                 execution_timeout=timedelta(minutes=5),
                 *args,
                 **kwargs):

        if env_vars is None:
            env_vars = {}

        envs = {
            "OUTPUT_FORMAT": output_format,
        }

        env_vars.update(envs)

        my_registry = "docker.io"
        my_project = "/tudracoon"

        super().__init__(
            dag=dag,
            image="{}{}/publictest:dcmConverter_53".format(my_registry, my_project), # Image from docker container /processing-container/dcm2itk_converter
            name="dcm2itk",
            env_vars=env_vars,
            image_pull_secrets=["registry-secret"],
            execution_timeout=execution_timeout,
            ram_mem_mb=1000,
            *args,
            **kwargs
            )
