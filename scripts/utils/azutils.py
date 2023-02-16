import numpy as np
import json

import azureml.core as azure_core
from azure.storage.blob import BlobServiceClient

def download_datasets(ws, dataset_name, data_download_path):
    raw_data_files = azure_core.Dataset.get_by_name(ws, name=dataset_name,)
    print("RAW DATA FILES ", raw_data_files)
    raw_data_files.download(target_path=data_download_path, overwrite=False)
    return

def download_from_blobstorage():
    STORAGEACCOUNTURL = "storageexplorer://v=1&accountid=/subscriptions/4053f59e-e58c-425d-90d1-e738fb166047/resourceGroups/sb-ml-dev/providers/Microsoft.Storage/storageAccounts/poimachinelearning&subscriptionid=4053f59e-e58c-425d-90d1-e738fb166047"
    STORAGEACCOUNTKEY = "p4EEQkYTnq4jkfVtyAC2iKhVwGRSP96AumqVC8YXAzxU6h3r3Ns/5L5FuKqm3R4WtxgfPyOhfC+3lGnSgimQSA=="
    CONTAINERNAME = "bfp-detection"
    BLOBNAME = "data"

    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
    blob_client_instance = blob_service_client_instance.get_blob_client(
        CONTAINERNAME, BLOBNAME, snapshot=None)
    blob_data = blob_client_instance.download_blob()
    data = blob_data.readall()


