# RACOON Dashboard

The full sourcecode of the dashboard website can be found in the 'files' directory

## Quick-Start Guide

This quickstart guide describes the steps to use the dashboard as a demo application with placeholder data inside a doccker container. Note that the demo data ranges from January 2021 to April 2021.

1. Install Docker on your machine
2. Open a terminal and navigate to the directory where the Dockerfile (and also this README) lies in.
3. Build the docker container with the following command. Don't forget the point at the end of the command. You can replace <name> with an arbitrary name for the docker container e.g. racoon_dashboard
```
docker build -t <name> .
```
4. Wait until the Container is built. This can take some minutes since the whole Python environment and all required pip packages will be downloaded
5. Start the docker container. ```-p XXXX:YYYY``` means that port YYYY of the docker container will be mapped to port XXXX of your host PC
```
docker run -p 8000:8000 -i -t <name>
```
6. Open the dashboard in your browser
     - Main Page: http://127.0.0.1:8000
     - Admin View: http://127.0.0.1:8000/admin (User: admin  - Password: admin)
     - API View: http://127.0.0.1:8000/api (Note that some API views are very large and may take some time to load since they show all available data on one page)


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

