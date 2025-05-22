import os
import json
import requests
import pandas as pd
from zipfile import ZipFile

def download(cache):
    dataset_path = os.path.join(cache, "hrrsd")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URL for the Kaggle dataset
    url = "https://www.kaggle.com/api/v1/datasets/download/haashaatif/hrrsd-dataset"
    zip_path = os.path.join(dataset_path, "archive.zip")
    
    # Configure Kaggle API authentication
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
    
    # Remove the zip file
    os.remove(zip_path)

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache):
    dataset_path = os.path.join(cache, "hrrsd")
    annotations_path = os.path.join(dataset_path, "Annotations.csv")
    classes_path = os.path.join(dataset_path, "Classes.csv")
    
    # Read the CSV files
    annotations_df = pd.read_csv(annotations_path)
    classes_df = pd.read_csv(classes_path)
    
    # Create a dictionary to store image data
    dataset = {}
    
    # Process each image's annotations
    for filename in annotations_df['filename'].unique():
        image_annotations = annotations_df[annotations_df['filename'] == filename]
        
        # Get image dimensions for normalization
        width = image_annotations['width'].iloc[0]
        height = image_annotations['height'].iloc[0]
        
        # Initialize image entry
        image_path = f"hrrsd/JpegImages/{filename}"
        dataset[image_path] = {
            "image_id": filename,
            "image_source": "hrrsd",
            "bboxes": [],
            "captions": ["An aerial image."],  # This dataset doesn't include captions
            "QA": []        # This dataset doesn't include QA pairs
        }
        
        # Process each bounding box
        for _, row in image_annotations.iterrows():
            # Normalize coordinates to 0-1 range
            x1 = row['xmin'] / width
            y1 = row['ymin'] / height
            x2 = row['xmax'] / width
            y2 = row['ymax'] / height
            
            # Add normalized bounding box with label
            bbox = [row['class'], x1, y1, x2, y2]
            dataset[image_path]["bboxes"].append(bbox)
    
    print(f"Loaded hrrsd")
    return dataset

def info():
    return {
        "name": "TGRSHRRSD",
        "description": "Traffic Guide Road Sign Recognition and Self-Driving Dataset",
        "size": "2.8 MB (annotations only)",
        "num_datapoints": 21761,  # From unique values count in annotations
        "num_images": None,  # Would need to count unique filenames
        "publication_year": None,  # Information not provided
        "requires_gpt_conversion": False
    }