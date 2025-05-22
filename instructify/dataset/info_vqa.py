import os
import json
import io
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

def download(cache):
    """Process the InfoVQA dataset."""
    dataset_path = os.path.join(cache, "info_vqa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading InfoVQA dataset...")
    dataset = load_dataset("vidore/infovqa_train")
    
    # Prepare directories
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Process data and save images
    processed_data = {}
    errors = []
    
    print("Processing data and saving images...")
    for split in dataset.keys():
        for idx, example in tqdm(enumerate(dataset[split])):
            # Use the image filename as image_id (without the extension)
            image_id = os.path.splitext(example['image_filename'])[0]
            
            # Save image
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                image = example['image'].convert("RGB")
                image.save(image_path, quality=95)
            
            # Create QA pair
            question = example['query'].strip()
            answer = example['answer'].strip()
            qa_pair = f"Question: {question} Answer: {answer}"
            
            # Store processed data
            image_path = f"info_vqa/images/{image_id}.jpg"
            if image_path not in processed_data:
                processed_data[image_path] = {
                    'image_id': image_id,
                    'captions': [],  # No captions in this dataset
                    'bboxes': [],    # No bboxes in this dataset
                    'qa_pairs': [qa_pair],
                }
            else:
                processed_data[image_path]['qa_pairs'].append(qa_pair)
    
    # Save processed data
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    with open(processed_json_path, 'w') as f:
        json.dump(processed_data, f)
    
    print(f"Processed {len(processed_data)} items")
    if errors:
        print(f"Encountered {len(errors)} errors")
        error_path = os.path.join(dataset_path, "processing_errors.json")
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
    
    # Create processed flag
    with open(processed_flag, 'w') as f:
        f.write("")

def load(cache):
    """Load the processed dataset."""
    dataset_path = os.path.join(cache, "info_vqa")
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    info_vqa_data = {}
    
    for image_path, data in processed_data.items():
        info_vqa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "info_vqa",
            "captions": data['captions'],
            "bboxes": data['bboxes'],
            "QA": data['qa_pairs'],
        }
    
    return info_vqa_data

def info():
    """Return information about the InfoVQA dataset."""
    return {
        "name": "InfoVQA",
        "description": "Information Visual Question Answering dataset",
        "size": "10K-100K",  # Based on the dataset card
        "num_datapoints": 10074,  # Based on 10.1k rows shown
        "num_images": 2118,     # Assuming one image per datapoint
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }