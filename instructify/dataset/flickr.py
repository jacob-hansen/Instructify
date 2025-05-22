import os
import json
import requests
from zipfile import ZipFile

# Requires Kaggle Authentication
#   Get key from Kaggle account settings
#   It will download a Kaggle.json of the form {"username":"xxx","key":"xxx"}
#   Place at ~/.kaggle/kaggle.json

def download(cache):
    dataset_path = os.path.join(cache, "flickr")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URL for the Kaggle dataset
    url = "https://www.kaggle.com/api/v1/datasets/download/hsankesara/flickr-image-dataset"
    zip_path = os.path.join(dataset_path, "archive.zip")
    
    # Configure Kaggle API authentication
    # You need to have your Kaggle API token in ~/.kaggle/kaggle.json
    try:
        with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
            api_token = json.load(f)
            headers = {
                "Authorization": f"Basic {api_token['key']}"
            }
    except FileNotFoundError:
        raise Exception("Please set up your Kaggle API credentials in ~/.kaggle/kaggle.json")

    # Download the dataset
    print("Downloading dataset...")
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download dataset: {response.status_code}")

    # Unzip the downloaded file
    print("Extracting dataset...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    
    # Get results.csv
    os.rename(os.path.join(dataset_path, "flickr30k_images/results.csv"), os.path.join(dataset_path, "results.csv"))
    os.rmdir(os.path.join(dataset_path, "flickr30k_images"))

    # Remove the zip file
    os.remove(zip_path)

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache):
    dataset_path = os.path.join(cache, "flickr")
    annotations_path = os.path.join(dataset_path, "results.csv")
    
    # Create a dictionary to store image captions
    flickr_data = {}
    
    # Read and parse the results.csv file
    unique_images = set()
    with open(annotations_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in f:
            # Split by | and strip whitespace from each part
            parts = [part.strip() for part in line.strip().split('|')]
            if len(parts) < 3:  # We expect at least image_name, index, and caption
                print(f"Skipping malformed line: {line.strip()}")
                continue
                
            image_name = parts[0]
            caption = parts[2]  # The caption is the third part after splitting
            
            image_path = "flickr30k/" + image_name
            unique_images.add(image_path)
            
            if image_path not in flickr_data:
                flickr_data[image_path] = {
                    "image_id": image_name,
                    "image_source": "flickr30k",
                    "bboxes": [],  # Flickr dataset doesn't include bounding boxes
                    "captions": [],
                    "QA": []       # No QA pairs in this dataset
                }
            flickr_data[image_path]["captions"].append(caption)
            
    print(f"Loaded flickr")
    return flickr_data

def info():
    return {
        "name": "Flickr30k",
        "description": "Flickr30k dataset",
        "size": "4.4 GB",
        "num_datapoints": 158914, 
        "num_images": 31783,
        "publication_year": 2015,
        "requires_gpt_conversion": False
    }