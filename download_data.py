import os, sys, glob
import ibm_boto3
from ibm_botocore.client import Config

BUCKET='w251-final-project'
auth_endpoint = 'https://iam.bluemix.net/oidc/token'
service_endpoint = 'https://s3.us-south.cloud-object-storage.appdomain.cloud'

creds = {
  "apikey": "f-qBpDnSKCQy-09Qp0ITDMqtuAKL3dw_ceFqbl2plO9z",
  "cos_hmac_keys": {
    "access_key_id": "81125ccd0720474695df163a3d531386",
    "secret_access_key": "022d50955047be30956d7c7c0600d1e911baf16b877ce7bf"
  },
  "endpoints": "https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints",
  "iam_apikey_description": "Auto-generated for key 81125ccd-0720-4746-95df-163a3d531386",
  "iam_apikey_name": "Service credentials-1",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Reader",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/37d593cd072c45d790ab0474d61bb350::serviceid:ServiceId-f507b0a6-adda-4fd4-ae74-8072bcac69cc",
  "resource_instance_id": "crn:v1:bluemix:public:cloud-object-storage:global:a/37d593cd072c45d790ab0474d61bb350:f7e02bf8-aadf-4b08-bcdd-62e88994d0a8::"
}

resource = ibm_boto3.resource('s3',
                ibm_api_key_id=creds['apikey'],
                ibm_service_instance_id=creds['resource_instance_id'],
                ibm_auth_endpoint=auth_endpoint,
                config=Config(signature_version='oauth'),
                endpoint_url=service_endpoint)

if __name__ == "__main__":
    os.mkdir('split_data')
    os.mkdir('split_data/train')
    os.mkdir('split_data/test')
    os.mkdir('split_data/validation')

    # get list of objects from the bucket
    client = resource.meta.client
    page = client.get_paginator('list_objects')
    for page in page.paginate(Bucket=BUCKET):
        keys = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
        if keys:
            for k in keys:
                key = k['Key']
                path = 'split_data/' + '/'.join(key.split('/')[:-1])
                if not os.path.exists(path):
                    os.mkdir(path)
                # download key to path
                filename = 'split_data/' + key
                print(filename)
                client.download_file(Bucket=BUCKET, Key=key, Filename=filename)
