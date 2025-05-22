import os
import csv
import requests
from zipfile import ZipFile

def download_and_extract_zip(url, zip_path, extract_dir):
    print(f"Downloading {os.path.basename(zip_path)}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Extracting {os.path.basename(zip_path)}...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up zip file
        os.remove(zip_path)
    else:
        raise Exception(f"Failed to download {os.path.basename(zip_path)}: {response.status_code}")
        
def download(cache):
    dataset_path = os.path.join(cache, "remoteclip_det10")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URLs for the RemoteCLIP Det-10 dataset
    csv_url = "https://huggingface.co/datasets/gzqy1026/RemoteCLIP/resolve/main/csv_file/train/Det-10.csv"
    images_url_1 = "https://huggingface.co/datasets/gzqy1026/RemoteCLIP/resolve/main/data/Det-10_part1.zip"
    images_url_2 = "https://huggingface.co/datasets/gzqy1026/RemoteCLIP/resolve/main/data/Det-10_part2.zip"
    
    csv_path = os.path.join(dataset_path, "Det-10.csv")
    images_zip_path_1 = os.path.join(dataset_path, "Det-10_part1.zip")
    images_zip_path_2 = os.path.join(dataset_path, "Det-10_part2.zip")
    images_dir = os.path.join(dataset_path, "images")

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Download CSV file
    print("Downloading CSV annotations...")
    response = requests.get(csv_url)
    if response.status_code == 200:
        with open(csv_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download CSV: {response.status_code}")

    # Download and extract both zip files
    download_and_extract_zip(images_url_1, images_zip_path_1, images_dir)
    download_and_extract_zip(images_url_2, images_zip_path_2, images_dir)

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache):
    dataset_path = os.path.join(cache, "remoteclip_det10")
    csv_path = os.path.join(dataset_path, "Det-10.csv")
    
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

                image_path = os.path.join("remoteclip_det10/images", image_filename)
                if image_path not in remoteclip_data:
                    remoteclip_data[image_path] = {
                        "image_id": image_filename,
                        "image_source": "remoteclip_det10",
                        "bboxes": [],  # RemoteCLIP doesn't include bounding boxes
                        "captions": ["An aerial image."],
                        "QA": []      # No QA pairs in this dataset
                    }
                
                remoteclip_data[image_path]["captions"].append(caption.strip())
        
            if image_path in remoteclip_data:
                remoteclip_data[image_path]["captions"] = list(set(remoteclip_data[image_path]["captions"]))  # Remove duplicates
    
    return remoteclip_data

def info():
    return {
        "name": "RemoteCLIP Det-10",
        "description": "A dataset for remote sensing image detection and captioning, containing aerial/satellite images with corresponding human-written captions.",
        "size": "2.2 GB",
        "num_datapoints": 70052,
        "num_images": 13718,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }