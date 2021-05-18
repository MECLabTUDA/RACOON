# RACOON JIP Workflows

For using these workflows, a running JIP installation is required. Follow the guide in the JIP_Guide.pdf to install the platform on a local test machine.

## Workflow Installation

1. Move all contents from **workflows/airflow-components** to **FAST_DATA_DIR/workflows/**. FAST_DATA_DIR is the directory that was selected during JIP installation.
2. Currently all Docker Containers have been pushed to Dockerhub. If your machine has internet access, there is no need to rebuild any containers since they will be automatically pulled by the workflows.
3. If you want to **rebuild the Docker containers**, they can be found in **workflows/processing-container**. After building them you have to modify the corresponding operators that they know where the containers can be found. The Operators can be found in **workflows/airflow-compnents/dags/tuda**. Here is a short list of which Docker Container belongs to which Operator.

| Operator                   | Processing Container |
| -------------------------- |:--------------------:| 
| Dcm2ItkOperator            | dcm2itk_converter    |
| DcmSr2JsonOperator.py      | jsonDcmSr_tools      |
| Json2DcmSrOperator.py      | jsonDcmSr_tools      |
| QmArtifactsOperator.py     | qm_artifacts         |
| QmDicePredictorOperator.py | qm_dicePredictor     |
