import os
import os
import json
import zipfile
import tarfile
import requests
from collections import defaultdict

def download(cache):
    # Create the directory for LVIS dataset
    lvis_dir = os.path.join(cache, 'lvis')
    os.makedirs(lvis_dir, exist_ok=True)
    
    # Check if the dataset is already downloaded
    downloaded_file_path = os.path.join(lvis_dir, "downloaded")
    if os.path.exists(downloaded_file_path):
        print("Dataset already downloaded.")
        return

    # URLs for downloading LVIS and ImageNet datasets
    lvis_url = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"

    # Download LVIS dataset
    lvis_zip_path = os.path.join(lvis_dir, 'lvis_v1_train.json.zip')
    if not os.path.exists(lvis_zip_path):
        print("Downloading LVIS dataset...")
        response = requests.get(lvis_url)
        with open(lvis_zip_path, 'wb') as f:
            f.write(response.content)

    # Unzip LVIS dataset
    with zipfile.ZipFile(lvis_zip_path, 'r') as zip_ref:
        zip_ref.extractall(lvis_dir)
    os.remove(lvis_zip_path)  # Remove zip file after extraction

    # Create the downloaded marker file
    open(downloaded_file_path, 'w').close()
    print("Download completed.")

def load(cache):
    lvis_dir = os.path.join(cache, 'lvis')
    lvis_file = os.path.join(lvis_dir, 'lvis_v1_train.json')

    # Load the entire file content as a string
    with open(lvis_file, 'r') as f:
        lvis_data = f.read()

    # Extract the 'annotations' and 'images' using string operations to speed up (40s vs 13s)
    annotations_base_string = lvis_data.split('"annotations": ', 1)[1].split(', "images"')[0]
    annotations_clean = "}, {".join([ann.split(', "segmentation"')[0] for ann in annotations_base_string.split("}, {")]) + "}]"
    annotations_json = json.loads(annotations_clean)
    images_base_string = lvis_data.split('"images": ', 1)[1].split(', "licenses"')[0]
    images_json = json.loads(images_base_string)

    # Create a mapping for categories by their ID for easy lookup
    categories_base_string = lvis_data.split('"categories": ', 1)[1][:-1]
    categories_dict = {cat['id']: cat['name'].replace('_', ' ') for cat in json.loads(categories_base_string)}
    
    # Initialize the dictionary to store image metadata
    lvis_image_metadata = {}

    # Create a mapping from image_id to annotations
    annotations_dict = defaultdict(list)
    for ann in annotations_json:
        annotations_dict[ann['image_id']].append(ann)

    # Iterate over each image entry
    for image_info in images_json:
        image_id = image_info['coco_url'].split('/')[-1].split('.')[0]
        image_path = "coco/train2017/" + image_info['coco_url'].split('/')[-1]
        
        # Retrieve annotations for the current image using the mapping
        annotations = annotations_dict[image_info['id']]

        # Collect category information for the annotations
        categories = [categories_dict[ann['category_id']] for ann in annotations]

        # Collect negative categories from images_json["neg_category_ids"] (for a caption)
        neg_categories = [categories_dict[cat_id] for cat_id in image_info["neg_category_ids"]]
        captions = []
        if len(neg_categories) > 0:
            captions.append(f"This image does not contain {', '.join(neg_categories)}.")
        if len(categories) > 0:
            captions.append(f"This image contains {', '.join(set(categories))}.")
        
        # Get the image dimensions
        image_width = image_info['width']
        image_height = image_info['height']

        # Convert from [x, y, w, h] -> [x1, y1, x2, y2] and normalize the bounding boxes
        normalized_bboxes = [
            [categories[i]] + [
                ann['bbox'][0] / image_width,
                ann['bbox'][1] / image_height,
                (ann['bbox'][0] + ann['bbox'][2]) / image_width,
                (ann['bbox'][1] + ann['bbox'][3]) / image_height
            ] for i, ann in enumerate(annotations)
        ]

        # Create the metadata structure for this image
        lvis_image_metadata[image_path] = {
            "image_id": str(image_id),
            "image_source": "lvis",
            "bboxes": normalized_bboxes,
            "captions": captions,
            "QA": []  # No QA
        }
    
    return lvis_image_metadata

def info():
    return {
        "name": "LVIS Dataset",
        "description": "A large-scale dataset for training object detection models.",
        "size": "1100 MB",
        "num_datapoints": 1470481,
        "num_images": 1000,
        "publication_year": 2019,
        "requires_gpt_conversion": False
    }
