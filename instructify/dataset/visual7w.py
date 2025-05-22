import os
import json
import zipfile
import requests
from utils import image_id_mapping

def download(cache):
    dataset_name = "visual7w"
    dataset_folder = os.path.join(cache, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_folder, "downloaded")
    if os.path.exists(downloaded_flag):
        print(f"{dataset_name} is already downloaded.")
        return
    
    # URL and paths
    url = "https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
    zip_path = os.path.join(dataset_folder, "dataset_v7w_telling.zip")
    json_path = os.path.join(dataset_folder, "dataset_v7w_telling.json")
    
    # Download the dataset
    print(f"Downloading {dataset_name}...")
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the specific JSON file
    print(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract('dataset_v7w_telling.json', dataset_folder)
    
    # Clean up the zip file
    os.remove(zip_path)
    
    # Create a 'downloaded' flag
    open(downloaded_flag, 'w').close()
    print(f"{dataset_name} downloaded and extracted successfully.")

def load(cache):
    dataset_name = "visual7w"
    dataset_folder = os.path.join(cache, dataset_name)
    json_path = os.path.join(dataset_folder, "dataset_v7w_telling.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist. Please run download() first.")

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)["images"]

    # Filter data to only contain "train" split
    data = [d for d in data if d["split"] == "train"]
    visual_genome_image_mapping = image_id_mapping(cache)

    visual7w_dataset = {}
    for group in data:
        image_id = group["image_id"]
        for row in group["qa_pairs"]:
            image_path = visual_genome_image_mapping[image_id]
            question = row["question"]
            answer = row["answer"].lower()
            options = [i.lower() for i in row["multiple_choices"]]
            
            # remove '.' from answer or options if it is the last character
            if answer[-1] == '.':
                answer = answer[:-1]
            for i in range(len(options)):
                if options[i][-1] == '.':
                    options[i] = options[i][:-1]
            if image_path not in visual7w_dataset:
                visual7w_dataset[image_path] = {
                        "image_id": image_id,
                        "image_source": image_path.split("/")[0],
                        "bboxes": [],     # No Boxes
                        "captions": [],  # No captions
                        "QA": []         
                    }
            # add QA, and negatives from "multiple_choices" field
            visual7w_dataset[image_path]["QA"].append(question + " " + answer + " (not " + ", nor ".join(options) + ")")

    return visual7w_dataset

def info():
    return {
        "name": "visual7w",
        "description": "The Visual7W dataset, a large-scale visual question answering dataset.",
        "size": "47 MB",
        "num_datapoints": 69817,
        "num_images": 14366,
        "publication_year": 2016,
        "requires_gpt_conversion": True
    }
