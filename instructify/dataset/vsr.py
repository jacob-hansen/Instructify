import os
import json
import requests

def download(cache):
    dataset_name = "vsr"
    dataset_folder = os.path.join(cache, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_folder, "downloaded")
    if os.path.exists(downloaded_flag):
        print(f"{dataset_name} is already downloaded.")
        return
    
    # URL and paths
    url = "https://github.com/cambridgeltl/visual-spatial-reasoning/blob/master/data/data_files/all_vsr_validated_data.jsonl?raw=true"
    jsonl_path = os.path.join(dataset_folder, "all_vsr_validated_data.jsonl")
    
    # Download the JSONL file
    print(f"Downloading {dataset_name}...")
    response = requests.get(url)
    with open(jsonl_path, 'wb') as f:
        f.write(response.content)
    
    # Create a 'downloaded' flag
    open(downloaded_flag, 'w').close()
    print(f"{dataset_name} downloaded successfully.")

def load(cache):
    dataset_name = "vsr"
    dataset_folder = os.path.join(cache, dataset_name)
    jsonl_path = os.path.join(dataset_folder, "all_vsr_validated_data.jsonl")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"{jsonl_path} does not exist. Please run download() first.")
    
    # Load the JSONL data and structure it into the required format
    formatted_data = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            image_id = data['image'].split('.')[0]  # Extract image ID from the filename
            captions = [data['caption']]
            image_path = f"coco/train2017/{image_id.zfill(12)}.jpg"
            if image_path not in formatted_data:
                formatted_data[image_path] = {
                    "image_id": image_id,
                    "image_source": "coco/train2017",
                    "bboxes": [],  # No bounding boxes in this dataset
                    "captions": captions,
                    "QA": []       # No QA pairs in this dataset
                }
            else:
                formatted_data[image_path]['captions'].extend(captions)
    return formatted_data

def info():
    return {
        "name": "vsr",
        "description": "The Visual Spatial Reasoning dataset, which includes image captions with spatial relations.",
        "size": "3 MB",
        "num_datapoints": 10972,
        "num_images": 6259,
        "publication_year": 2020,
        "requires_gpt_conversion": False
    }
