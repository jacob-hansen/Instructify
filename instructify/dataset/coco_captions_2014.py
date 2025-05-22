import os
import os
import re
import json
import requests
from zipfile import ZipFile

def download(cache):
    dataset_path = os.path.join(cache, "coco_captions_2014")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # URL for the COCO captions dataset
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    zip_path = os.path.join(dataset_path, "annotations_trainval2014.zip")
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # Download the dataset
    print("Downloading dataset...")
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

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
    dataset_path = os.path.join(cache, "coco_captions_2014")
    captions_file = os.path.join(dataset_path, "annotations", "captions_train2014.json")
    instances_file = os.path.join(dataset_path, "annotations", "instances_train2014.json")

    with open(captions_file) as f:
        coco_captions_total_data = json.load(f)
        coco_captions_annotations = coco_captions_total_data["annotations"]

    # Load instances file as text and remove the "segmentation" sections using regex
    with open(instances_file) as f:
        instances_data_text = f.read()

    # Regular expression to remove the "segmentation": [...] sections
    instances_data_text = re.sub(r'"segmentation": \[\[.*?\]\],?', '', instances_data_text)

    # Parse the cleaned-up text as JSON
    coco_instances_data = json.loads(instances_data_text)
    coco_instances_annotations = coco_instances_data["annotations"]

    # Create a mapping from category ID to category name
    category_id_to_name = {category['id']: category['name'] for category in coco_instances_data['categories']}

    # Create a mapping from image ID to image path
    coco_captions_image_id_mapping = {}
    coco_captions_image_size_mapping = {}
    for row in coco_captions_total_data["images"]:
        coco_captions_image_id_mapping[row["id"]] = "coco/train2017/" + row["file_name"].split("_")[-1]
        coco_captions_image_size_mapping[row["id"]] = (row["width"], row["height"])

    coco_captions = {}

    # Load captions
    for row in coco_captions_annotations:
        image_id = row["image_id"]
        image_path = coco_captions_image_id_mapping[image_id]
        if image_path not in coco_captions:
            coco_captions[image_path] = {
                "image_id": str(image_id),
                "image_source": "coco/train2014",
                "bboxes": [],
                "captions": [],
                "QA": []       # No QA
            }
        coco_captions[image_path]["captions"].append(row["caption"])

    # Load bounding boxes with category names
    for row in coco_instances_annotations:
        image_id = row["image_id"]
        image_path = coco_captions_image_id_mapping.get(image_id)

        if image_path:
            width, height = coco_captions_image_size_mapping[image_id]
            x, y, w, h = row["bbox"]

            # Normalize the bounding box coordinates
            x1 = x / width
            y1 = y / height
            x2 = (x + w) / width
            y2 = (y + h) / height

            # Get the category name from the ID
            category_name = category_id_to_name.get(row["category_id"], "unknown")

            coco_captions[image_path]["bboxes"].append([category_name, x1, y1, x2, y2])

    return coco_captions

def info():
    return {
        "name": "COCO Captions",
        "description": "Captions created through crowdsourcing (e.g. Amaxon Mechanical Turk), with 5 captions per image from different annotators.",
        "size": "858 MB",
        "num_datapoints": 1019020,
        "num_images": 82783,
        "publication_year": 2014,
        "requires_gpt_conversion": False
    }