import os
import json
from datasets import load_dataset

def download(cache):
    dataset_name = "refcocoplus"
    dataset_folder = os.path.join(cache, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_folder, "downloaded")
    if os.path.exists(downloaded_flag):
        print(f"{dataset_name} is already downloaded.")
        return
    
    # Load dataset from Hugging Face
    print(f"Downloading {dataset_name} from Hugging Face...")
    dataset = load_dataset("jxu124/refcocoplus")

    # Extract and save only the required columns
    json_path = os.path.join(dataset_folder, "refcocoplus.json")
    filtered_data = []
    for item in dataset['train']:
        filtered_data.append({
            "image_id": str(item['image_id']),
            "raw_image_info": item['raw_image_info'],
            "image_path": item['image_path'],
            "bbox": item['bbox'],
            "captions": item['captions']
        })
    
    with open(json_path, 'w') as f:
        json.dump(filtered_data, f)

    # Create a 'downloaded' flag
    open(downloaded_flag, 'w').close()
    print(f"{dataset_name} downloaded and saved as JSON successfully.")

def load(cache):
    dataset_name = "refcocoplus"
    dataset_folder = os.path.join(cache, dataset_name)
    json_path = os.path.join(dataset_folder, "refcocoplus.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist. Please run download() first.")
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = {}
    for item in data:
        image_id = item['image_id']
        image_source = "refcocoplus"
        image_data = json.loads(item['raw_image_info'])
        image_width = image_data['width']
        image_height = image_data['height']
        image_path = f"coco/train2017/{item['image_path'].split('_')[-1]}"
        box = [item['bbox'][0] / image_width, item['bbox'][1] / image_height, item['bbox'][2] / image_width, item['bbox'][3] / image_height]
        bboxes = [[cap]+box for cap in item['captions']]
        if image_path in formatted_data:
            formatted_data[image_path]["bboxes"].extend(bboxes)
        else:
            formatted_data[image_path] = {
                "image_id": image_id,
                "image_source": image_source,
                "bboxes": bboxes,
                "captions": [],
                "QA": []
            }
    
    return formatted_data

def info():
    return {
        "name": "refcocoplus",
        "description": "The refcocoplus dataset, a large-scale visual grounding dataset.",
        "size": "22 MB",
        "num_datapoints": 48920,
        "num_images": 16992,
        "publication_year": 2014,
        "requires_gpt_conversion": True
    }
