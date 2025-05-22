import os
import csv
import requests
from zipfile import ZipFile

def download(cache):
    dataset_path = os.path.join(cache, "remoteclip_ret3")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URLs for the RemoteCLIP Ret-3 dataset
    csv_url = "https://huggingface.co/datasets/gzqy1026/RemoteCLIP/resolve/main/csv_file/train/Ret-3_train.csv"
    images_url = "https://huggingface.co/datasets/gzqy1026/RemoteCLIP/resolve/main/data/Ret-3_train.zip"
    
    csv_path = os.path.join(dataset_path, "Ret-3_train.csv")
    images_zip_path = os.path.join(dataset_path, "Ret-3_train.zip")
    images_dir = os.path.join(dataset_path, "images")

    # Download CSV file
    print("Downloading CSV annotations...")
    response = requests.get(csv_url)
    if response.status_code == 200:
        with open(csv_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download CSV: {response.status_code}")

    # Download and extract images
    print("Downloading image archive...")
    response = requests.get(images_url, stream=True)
    if response.status_code == 200:
        with open(images_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract images
        print("Extracting images...")
        with ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        
        # Clean up zip file
        os.remove(images_zip_path)
    else:
        raise Exception(f"Failed to download images: {response.status_code}")

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache):
    dataset_path = os.path.join(cache, "remoteclip_ret3")
    csv_path = os.path.join(dataset_path, "Ret-3_train.csv")
    
    remoteclip_data = {}
    
    # Load and process CSV file
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for group in csv_reader:
            image_filename = None
            for row in group[::-1]:
                info = row.split('\t')
                if len(info) == 1:
                    caption = info[0]
                else:
                    caption, image_filename = info
                    
                if caption.endswith(" ."):
                    caption = caption[:-2] + "."

                image_path = "remoteclip_ret3/images/" + image_filename
                if image_path not in remoteclip_data:
                    remoteclip_data[image_path] = {
                        "image_id": image_filename,
                        "image_source": "remoteclip_ret3",
                        "bboxes": [],  # RemoteCLIP doesn't include bounding boxes
                        "captions": ["An aerial image."],
                        "QA": []      # No QA pairs in this dataset
                    }
                
                remoteclip_data[image_path]["captions"].append(caption.strip())
        remoteclip_data[image_path]["captions"] = list(set(remoteclip_data[image_path]["captions"]))  # Remove duplicates
    
    return remoteclip_data

def info():
    return {
        "name": "RemoteCLIP Ret-3",
        "description": "A dataset for remote sensing image captioning, containing aerial/satellite images with corresponding human-written captions.",
        "size": "1.1 GB",
        "num_datapoints": 57137,
        "num_images": 21761,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }