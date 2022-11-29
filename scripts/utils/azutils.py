import numpy as np
import json
import azureml.core as azure_core


def download_datasets(ws, dataset_name, data_download_path):
    raw_data_files = azure_core.Dataset.get_by_name(ws, name=dataset_name)
    print("RAW DATA FILES ", raw_data_files)
    raw_data_files.download(target_path=data_download_path, overwrite=True)
    return
