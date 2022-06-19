import requests
import os
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.storage.blob import BlobServiceClient



apiUrl = "https://api.videoindexer.ai"
accountId = "3bf91975-2650-494b-812f-e39521bc1c0c"
location = "trial"
apiKey = "1b4fc47daf7f4a0da72214aaa54b30d6"

connect_str = "DefaultEndpointsProtocol=https;AccountName=techgig;AccountKey=RcPuGMrA8+Pq4EQlMp0yhu95QjWLBOPXnXI+hOdMTQRNi9wDzN+TVIMxGchSAozHpdqMRVJWUAj/+AStkoEjrg==;EndpointSuffix=core.windows.net"
container_name = "videoanalytics"

def upload_blob(local_file_path,container_name,connect_str):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    filename = local_file_path.split(os.sep)[-1]
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    # Upload the created file
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data,overwrite=True)
    return f"https://techgig.blob.core.windows.net/{container_name}/{filename}"


connect_str = "DefaultEndpointsProtocol=https;AccountName=techgig;AccountKey=RcPuGMrA8+Pq4EQlMp0yhu95QjWLBOPXnXI+hOdMTQRNi9wDzN+TVIMxGchSAozHpdqMRVJWUAj/+AStkoEjrg==;EndpointSuffix=core.windows.net"
container_name = "videoanalytics"

local_file_path = "/Users/skj/Projects/Techgig/SBI/Video Analytics/source/data/face.mp4"
name = "video_analytics"


#API details
url = f"https://api.videoindexer.ai/Auth/{location}/Accounts/{accountId}//AccessTokenWithPermission?permission=Owner"
headers = {'Cache-Control': 'no-cache','Ocp-Apim-Subscription-Key':apiKey}

#Making http post request
response = requests.get(url, headers=headers, verify=False)
access_token = response.json()

# video_id = None # uncomment this if u want to upload a new video!
video_id = "0ff6ce15cf" # already uploaded and processed
if not video_id:   
    video_url = upload_blob(local_file_path,container_name,connect_str)

    
    #API details
    url = f"https://api.videoindexer.ai/{location}/Accounts/{accountId}/Videos?name={name}&videoUrl={video_url}&accessToken={access_token}"
    headers = {'Cache-Control': 'no-cache','Ocp-Apim-Subscription-Key':apiKey}

    #Making http post request
    response = requests.post(url, headers=headers,data=None, verify=False)

    video_id = response.json()["id"]

else:
    #API details
    url = f"https://api.videoindexer.ai/{location}/Accounts/{accountId}/Videos/{video_id}/Index?language=en-GB&reTranslate=false&includeStreamingUrls=true&includeSummarizedInsights=true&accessToken={access_token}"
    headers = {'Cache-Control': 'no-cache','Ocp-Apim-Subscription-Key':apiKey}

    #Making http post request
    response = requests.get(url, headers=headers,data=None, verify=False)

    for key,value in response.json()['summarizedInsights'].items():
        print(key)
        print(value)
        print()

