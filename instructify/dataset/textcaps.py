import os
import json
import requests
from zipfile import ZipFile

def download(cache):
    dataset_path = os.path.join(cache, "textcaps")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URLs for the TextCaps dataset
    annotations_url = "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json"
    annotations_path = os.path.join(dataset_path, "TextCaps_0.1_train.json")

    # Download annotations
    print("Downloading annotations...")
    response = requests.get(annotations_url)
    if response.status_code == 200:
        with open(annotations_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download annotations: {response.status_code}")

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache):
    dataset_path = os.path.join(cache, "textcaps")
    annotations_path = os.path.join(dataset_path, "TextCaps_0.1_train.json")
    
    # Load annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    textcaps_data = {}
    
    # Process annotations
    for item in data['data']:
        image_id = item['image_id']
        image_path =  f"textvqa/train_images/{image_id}.jpg"
        
        added_captions = set()
        if image_path not in textcaps_data:
            textcaps_data[image_path] = {
                "image_id": str(image_id),
                "image_source": "textcaps",
                "bboxes": [],  # TextCaps doesn't include bounding boxes
                "captions": [],
                "QA": []      # No QA pairs in this dataset
            }
        
        # Add main caption
        textcaps_data[image_path]["captions"].append(item['caption_str'])
        
        # Add reference captions
        textcaps_data[image_path]["captions"].extend(item['reference_strs'])
        
        # Add formatted reference tokens
        for ref_tokens in item['reference_tokens']:
            # Remove special tokens (<s> and </s>)
            cleaned_tokens = [token for token in ref_tokens if token not in ['<s>', '</s>']]
            # Join tokens and format
            text = ' '.join(cleaned_tokens)
            formatted_text = f"Text visible in image {text}"
            if formatted_text not in added_captions:
                added_captions.add(formatted_text)
                textcaps_data[image_path]["captions"].append(formatted_text)

        
    return textcaps_data

def info():
    return {
        "name": "TextCaps",
        "description": "A dataset for image captioning with a focus on text in images. Contains images with text and corresponding human-written captions that describe both the scene and the text content.",
        "size": "~7 GB",
        "num_datapoints": 14579,
        "num_images": 14575,
        "publication_year": 2017,
        "requires_gpt_conversion": False
    }