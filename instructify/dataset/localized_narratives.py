import os
import json
import requests
import pandas as pd

LOCALIZED_NARRATIVE_GROUPS = {
    "open_images": "open_images_train_v6_captions.jsonl", 
    "coco/train2017": "coco_train_captions.jsonl",
    "flickr30k": "flickr30k_train_captions.jsonl",
    "ade20k": "ade20k_train_captions.jsonl"
}
def download(cache):
    # Directory where the dataset files will be stored
    dataset_dir = os.path.join(cache, "localized_narratives")
    os.makedirs(dataset_dir, exist_ok=True)

    # URLs for the dataset files
    urls = [
        "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl",
        "https://storage.googleapis.com/localized-narratives/annotations/coco_train_captions.jsonl",
        "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_train_captions.jsonl",
        "https://storage.googleapis.com/localized-narratives/annotations/ade20k_train_captions.jsonl"
    ]

    # Download each file
    for url in urls:
        file_name = os.path.join(dataset_dir, url.split("/")[-1])
        if not os.path.exists(file_name):
            print(f"Downloading {url} to {file_name}...")
            response = requests.get(url)
            with open(file_name, "wb") as f:
                f.write(response.content)

    # Create a file to indicate the dataset has been downloaded
    with open(os.path.join(dataset_dir, "downloaded"), "w") as f:
        f.write("")

def load(cache):
    localized_narratives = {}
    for group, file_name in LOCALIZED_NARRATIVE_GROUPS.items():
        with open(os.path.join(cache, "localized_narratives", file_name), "r") as f:
            for line in f:
                image_information = json.loads(line)
                image_path = group + "/" + str(image_information["image_id"]).zfill(12) + ".jpg"
                if image_path not in localized_narratives:
                    localized_narratives[image_path] = {
                        "image_id": str(image_information["image_id"]),
                        "image_source": group,
                        "bboxes": [],  # No bounding boxes
                        "captions": [],
                        "QA": []       # No QA
                    }
                localized_narratives[image_path]["captions"].append(image_information["caption"])

    return localized_narratives


def info():
    return {
        "name": "Localized Narratives",
        "description": "Captions collected with human annotators for COCO, Open Images, Flickr30k, and ADE20K.",
        "size": "4750 MB",
        "num_data": 692738,
        "num_images": 672693,
        "publication_year": 2020,
        "requires_gpt_conversion": False
    }
