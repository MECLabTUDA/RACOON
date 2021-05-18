import os
import json
import glob
import pydicom
from os.path import join, basename, dirname
from datetime import timedelta
from dicomweb_client.api import DICOMwebClient
from multiprocessing.pool import ThreadPool
from kaapana.operators.HelperDcmWeb import HelperDcmWeb
from kaapana.operators.KaapanaPythonBaseOperator import KaapanaPythonBaseOperator
from kaapana.blueprints.kaapana_global_variables import BATCH_NAME, WORKFLOW_DIR


class GetAllQmDsrOperator(KaapanaPythonBaseOperator):
    def download_series(self, series):
        print("# Downloading series: {}".format(series["series_uid"]))
        try:
            download_successful = HelperDcmWeb.downloadSeries(
                seriesUID=series["series_uid"],
                target_dir=series['target_dir']
            )
            if not download_successful:
                exit(1)
            message = f"OK: Series {series['series_uid']}"
        except Exception as e:
            print(f"#### Something went wrong: {series['series_uid']}")
            print(e)
            message = f"ERROR: Series {series['series_uid']}"

        return message

    def get_files(self, ds, **kwargs):
        print("# Starting module LocalGetRefSeriesOperator")

        client = DICOMwebClient(url=self.pacs_dcmweb, qido_url_prefix="rs", wado_url_prefix="rs", stow_url_prefix="rs")

        run_dir = join(WORKFLOW_DIR, kwargs['dag_run'].run_id)
        download_series_list = []

        target_dir = join(run_dir, BATCH_NAME)
        print("#")
        print(f"# target_dir: {target_dir}")
        print("#")
        search_filters = {}
        search_filters['Modality'] = 'SR'
        search_filters['SeriesDescription'] = '*Single QM*'
        print("#")
        print("# Searching for series with the following filters:")
        print(json.dumps(search_filters, indent=4, sort_keys=True))
        print("#")
        pacs_series = client.search_for_series(search_filters=search_filters)
        print(f"Found series: {len(pacs_series)}")
        if len(pacs_series) == 0:
            print("# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("# ")
            print(f"No data found")
            print("# Abort.")
            print("# ")
            print("# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            exit(1)
        for series in pacs_series:
            series_uid = series['0020000E']['Value'][0]
            download_series_list.append(
                {
                    "series_uid": series_uid,
                    "target_dir": join(target_dir, series_uid, self.operator_out_dir)
                }
            )

        if len(download_series_list) == 0:
            print("# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("# ")
            print("# No series to download could be found!")
            print("# Abort.")
            print("# ")
            print("# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            exit(1)

        results = ThreadPool(self.parallel_downloads).imap_unordered(self.download_series, download_series_list)

        for result in results:
            print(result)
            if "error" in result.lower():
                exit(1)

    def __init__(self,
                 dag,
                 name='get-all-qm-sr',
                 parallel_downloads=3,
                 pacs_dcmweb_host='http://dcm4chee-service.store.svc',
                 pacs_dcmweb_port='8080',
                 aetitle="KAAPANA",
                 batch_name=None,
                 *args, **kwargs):

        self.pacs_dcmweb = pacs_dcmweb_host+":"+pacs_dcmweb_port + "/dcm4chee-arc/aets/"+aetitle.upper()
        self.parallel_downloads = parallel_downloads

        super().__init__(
            dag,
            name=name,
            batch_name=batch_name,
            python_callable=self.get_files,
            execution_timeout=timedelta(minutes=60),
            *args, **kwargs
        )
