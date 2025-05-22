import os
import json
import pandas as pd
import requests
from zipfile import ZipFile

def download(cache):
    # Directory where the dataset files will be stored
    dataset_dir = os.path.join(cache, "visual_genome")
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_dir, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URLs for the dataset files
    urls = [
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
    ]

    # Download and unzip each file
    for url in urls:
        file_name = os.path.join(dataset_dir, os.path.basename(url))
        if not os.path.exists(file_name.replace('.zip', '.json')):
            print(f"Downloading {url} to {file_name}...")
            response = requests.get(url)
            with open(file_name, "wb") as f:
                f.write(response.content)
            with ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            os.remove(file_name)

    # Create a file to indicate the dataset has been downloaded
    with open(os.path.join(dataset_dir, "downloaded"), "w") as f:
        f.write("")

def load(cache):
    dataset_dir = os.path.join(cache, "visual_genome")
    
    # Check if COCO annotations file exists
    coco_annotations_path = os.path.join(cache, "coco_captions", "annotations", "instances_train2017.json")
    if not os.path.exists(coco_annotations_path):
        raise FileNotFoundError("COCO annotations file not found. Please download coco_captions dataset first.")

    # Load COCO images data
    with open(coco_annotations_path, 'r') as f:
        coco_images_data = json.loads(f.read().split('"images": ', 1)[1].split(',"annotations":', 1)[0])
    
    coco_image_ids = {img['id']: img['file_name'] for img in coco_images_data}

    # Load JSON data
    with open(os.path.join(dataset_dir, 'objects.json')) as f:
        visual_genome_objects = json.load(f)
    with open(os.path.join(dataset_dir, 'attributes.json')) as f:
        visual_genome_attributes = json.load(f)
    with open(os.path.join(dataset_dir, 'relationships.json')) as f:
        visual_genome_relationships = json.load(f)
    with open(os.path.join(dataset_dir, 'image_data.json')) as f:
        visual_genome_image_data = json.load(f)

    visual_genome_info = {}

    vg_100k_path = os.path.join(cache, "images/vg/VG_100K")
    vg_100k_2_path = os.path.join(cache, "images/vg/VG_100K_2")
    assert os.path.exists(vg_100k_path), "Images for Visual Genome not found. Please download them."
    assert os.path.exists(vg_100k_2_path), "Images for Visual Genome not found. Please download them."
    vg_100k_images = os.listdir(vg_100k_path)
    vg_100k_image_mapping = {int(img.split(".")[0]): img for img in vg_100k_images}
    vg_100k_2_images = os.listdir(vg_100k_2_path)
    vg_100k_2_image_mapping = {int(img.split(".")[0]): img for img in vg_100k_2_images}

    # Create a mapping from image_id to image path
    visual_genome_images = {}
    for img in visual_genome_image_data:
        img_id = img["image_id"]
        if img["coco_id"] and img["coco_id"] in coco_image_ids:
            visual_genome_images[img_id] = {
                "image": f"coco/train2017/{coco_image_ids[img['coco_id']]}", 
                "width": img["width"], 
                "height": img["height"]
            }
        else:
            if img_id in vg_100k_image_mapping:
                visual_genome_images[img_id] = {
                    "image": f"vg/VG_100K/{vg_100k_image_mapping[img_id]}",
                    "width": img["width"], 
                    "height": img["height"]
                }
            elif img_id in vg_100k_2_image_mapping:
                visual_genome_images[img_id] = {
                    "image": f"vg/VG_100K_2/{vg_100k_2_image_mapping[img_id]}",
                    "width": img["width"], 
                    "height": img["height"]
                }
    
    # Create a mapping from object_id to object name
    visual_genome_info = {}
    for row in visual_genome_objects:
        image_id = int(row["image_id"])
        if image_id not in visual_genome_images:
            print(f"WARNING: Image {image_id} not found in image data, skipping.")
            continue

        image_path = visual_genome_images[image_id]["image"]
        objects = {}
        for obj in row["objects"]:
            bbox = [obj["x"], obj["y"], obj["x"] + obj["w"], obj["y"] + obj["h"]]
            # normalize by image width and height
            bbox = [bbox[0]/visual_genome_images[row["image_id"]]["width"], bbox[1]/visual_genome_images[row["image_id"]]["height"], bbox[2]/visual_genome_images[row["image_id"]]["width"], bbox[3]/visual_genome_images[row["image_id"]]["height"]]
            objects[obj["object_id"]] = {
                "category": obj["names"][0],
                "bbox": bbox,
                "attributes": []
            }
        if visual_genome_images[row["image_id"]]["image"] in visual_genome_info:
            visual_genome_info[visual_genome_images[row["image_id"]]["image"]] = objects
        else:
            visual_genome_info[visual_genome_images[row["image_id"]]["image"]] = objects

    # Add attributes to the objects
    for row in visual_genome_attributes:
        image_id = row["image_id"]
        if image_id not in visual_genome_images:
            continue
        image_path = visual_genome_images[image_id]["image"]
        for obj in row["attributes"]:
            if image_path in visual_genome_info:
                object_id = obj["object_id"]
                if object_id in visual_genome_info[image_path] and "attributes" in obj:
                    visual_genome_info[image_path][object_id]["attributes"].extend(obj["attributes"])

    # Add relationships to the objects
    for row in visual_genome_relationships:
        image_id = row["image_id"]
        if image_id not in visual_genome_images:
            continue
        image_path = visual_genome_images[image_id]["image"]
        
        relationships = row["relationships"]
        for rel_info in relationships:
            rel = rel_info["predicate"]
            subject_id = rel_info["subject"]["object_id"]
            object_name = rel_info["object"]["name"] if "name" in rel_info["object"] else rel_info["object"]["names"][0]
            if image_path in visual_genome_info:
                if subject_id in visual_genome_info[image_path]:
                    visual_genome_info[image_path][subject_id]["attributes"].append(f"{rel.lower()} the {object_name}")

    # cast attributes to set and then back to list
    for k, v in visual_genome_info.items():
        for obj_id, obj_info in v.items():
            obj_info["attributes"] = list(set(obj_info["attributes"]))

    # Construct the dataset
    visual_genome_dataset = {}
    for k, v in visual_genome_info.items():
        bboxes_formated = []
        for obj in v.values():
            # if len(obj["attributes"]) > 0:
            #     label_str = f"{obj['category']} ({', '.join(obj['attributes'])})"
            # else:
            #     label_str = obj["category"]
            # bboxes_formated.append([label_str] + obj["bbox"])
            bboxes_formated.append([obj['category']] + obj["bbox"])
            for attr in obj["attributes"]:
                bboxes_formated.append([attr] + obj["bbox"])
        visual_genome_dataset[k] = {
                    "image_id": k.split("/")[1].split(".")[0],
                    "image_source": k.split("/")[0],
                    "bboxes": bboxes_formated,
                    "captions": [],  # No captions
                    "QA": []         # No QA
                }
    return visual_genome_dataset

def info():
    return {
        "name": "Visual Genome",
        "description": "Visual Genome dataset containing objects, attributes, and relationships created by crowdsourcing.",
        "size": "1500 MB",
        "num_datapoints": 2510982,
        "num_images": 107787,
        "publication_year": 2016,
        "requires_gpt_conversion": False
    }
