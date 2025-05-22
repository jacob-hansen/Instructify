import os
import json
import requests
import pandas as pd
from zipfile import ZipFile

def download(cache):
    dataset_path = os.path.join(cache, "image2paragraph")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URL for the dataset
    url = "https://www.kaggle.com/api/v1/datasets/download/vakadanaveen/stanford-image-paragraph-captioning-dataset"
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
    
    # move stanford_img/content/stanford_images/ to images/
    stanford_images_path = os.path.join(dataset_path, "stanford_img/content/stanford_images")
    images_path = os.path.join(dataset_path, "images")
    os.rename(stanford_images_path, images_path)
    os.rmdir(os.path.join(dataset_path, "stanford_img/content"))
    os.rmdir(os.path.join(dataset_path, "stanford_img"))

    # Remove the zip file
    os.remove(zip_path)

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache):
    dataset_path = os.path.join(cache, "image2paragraph")
    data_file = os.path.join(dataset_path, "stanford_df_rectified.csv")
    
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    # Verify required columns exist
    required_columns = ['train', 'url', 'Paragraph']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter to include only training data
    train_df = df[df['train'] == True]
    
    # Create a dictionary to store image captions
    stanford_data = {}
    
    for _, row in train_df.iterrows():
        try:
            # Convert URL to image path
            img_name = row['url'].split('/')[-1]
            if "VG_100K_2" in row['url']:
                image_path = f"vg/VG_100K_2/{img_name}"
            elif "VG_100K" in row['url']:
                image_path = f"vg/VG_100K/{img_name}"
            else:
                raise ValueError(f"Unknown image source URL format: {row['url']}")
            
            if image_path not in stanford_data:
                stanford_data[image_path] = {
                    "image_id": os.path.basename(image_path),
                    "image_source": "vg" if image_path.startswith('vg') else "coco",
                    "bboxes": [],  # No bounding boxes in this dataset
                    "captions": [],
                    "QA": []      # No QA pairs in this dataset
                }
            
            # Add the paragraph as a caption
            stanford_data[image_path]["captions"].append(row['Paragraph'])
            
        except ValueError as e:
            print(f"Warning: Skipping row due to invalid URL format: {e}")
    
    print(f"Loaded image2paragraph")
    return stanford_data

def info():
    return {
        "name": "Stanford Image-Paragraph Captioning",
        "description": "Stanford Image-Paragraph Captioning dataset (training split)",
        "size": "Varies",
        "num_datapoints": 14579,
        "num_images": 14575,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }