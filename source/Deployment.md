# [SBI - Video Analytics](https://www.techgig.com/hackathon/video-analytics) 

## Scope of Work of the PoC

The POC curently is restricted to handling only 
- Cognitive Services to detect textual information from the frames (Video Indexer)
- Cognitive Services to detect objects (Custom YOLOv5 Models & Computer Vision - Fire & Weapon)
- Azure Functions for parallel processing of videos and scaling
- Custom model(DeepFace) to extract face and Face Analysis and Azure Face Detection 
- Azure Containers to store videos
    
Features - 
- Count of People
- Time taken for activity in premises
- Identification of known miscreants
- Identification of known facilitators
- Hazardous Objects being brought into the premises.
- Suspicious Activities
- Fire / Theft related incident taking place

Post POC Work - 
1. Train Custom Models with better domain data and camera views.
2. Increase Faces for known miscreants and facilitators in the database.
3. Creating a Azure Pipeline to read data in batches from containers, processing them at scale in parallel using Azure functions and storing the ouput json into DB.
4. Train the SSD on human data (with a top-down view).


## Pre-requisites from the Bank’s side
1. Azure Account
2. Works with both CPU and GPU
3. Setup & Deployment will not require more than a week.

## Infrastructure required for setting up the PoC. 
1. The Videos can be uploaded to a container storage.
2. An Azure function to deploy this code and call it using REST API's.
3. Azure Video Indexers 

## Setting up the PoC infrastructure on Microsoft Azure cloud setup
1. [Creating a resource Group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal#create-resource-groups)
2. [Creating a storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
3. [Creating a container](https://docs.microsoft.com/en-us/azure/storage/blobs/blob-containers-cli#create-a-container)
4. [Creating a function app](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-function-app-portal#create-a-function-app)
5. [Deploy code in azure function](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-function-app-portal#create-function)
6. [Writing scipts to download and upload images to the azure function for processing in azure](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=environment-variable-windows#upload-blobs-to-a-container)
7. [Creating a Video Indexer Resource](https://docs.microsoft.com/en-us/azure/azure-video-indexer/create-account-portal)
8. [Using Video Indexer APIs](https://docs.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-use-apis)

## High level PoC Key Performance Indicators (KPIs) 
- Samples Processed are provide in the [sample_data](./sample_data/) directory and Video of execution can be found [here](https://drive.google.com/file/d/1k1qvvdyT0Tu9BOUpNNaPHiwgscpePQe4/view?usp=sharing) 

- I have trained around 2 classes for object Detection using YOLOv5 
```
# YOLOv5 🚀 v6.1-258-g1156a32 Python-3.7.13 torch-1.11.0+cu113 CUDA:0 (Tesla T4, 15110MiB)

# hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
# optimizer: SGD with parameter groups 123 weight (no decay), 126 weight, 126 bias
    
# Image sizes 448 train, 448 val
# Starting training for 50 epochs...
# 50 epochs completed in 0.920 hours.

# Class     Images     Labels          P          R     mAP@.5 
#  all         50         86      0.927      0.901      0.939 
#  gun         50         13      0.961          1      0.995     
# fire         50         73      0.893      0.803      0.882      

# Results saved to runs/train/fire-theft
```
- The trainign Notebook can be found [here](./Detection%20Model%20Training%20YOLOv5.ipynb) and [dataset](https://app.roboflow.com/ds/azOMQ8HlGj?key=ZczUoFZvPLhttps://app.roboflow.com/ds/azOMQ8HlGj?key=ZczUoFZvPL) was annoted and created in YOLO format using [roboflow](https://app.roboflow.com/)

- Face Detection, Verification and Analysis for Frames were done using a custom local model [DeepFace](https://github.com/serengil/deepface) since calling Azure API for n no of pages would be costly in long run. 

- A directory was craeted to recognize faces from videos which can be found [here](./database/)

- Trained models can be found [here](./models/)

- People count Tracking was done using SSD and centroid tracking found [here](https://github.com/saimj7/People-Counting-in-Real-Time/)

## Deployment Guide
Local Deployment
- [Python](https://www.python.org/downloads/release/python-390/)
- Download yolo5 directory from [here](https://drive.google.com/file/d/1jHBCgqwypxS-EEpMWH7wrdEOch87OYwS/view?usp=sharing) and paste it in ./source/models/ and unzip it to use.
- First Run will install all the required custom models being used.
- The Credentails and keys provided with this POC will be avilable till the challenge duration

```cmd
pip3 install vitualenv
virtualenv vid_ana
source "vid_ana/bin/activate"
pip3 install -r requirements.txt
python face_detection.py
python other_detection.py
python people_counting.py
python azure_video_indexer.py
```


