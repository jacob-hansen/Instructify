import os
import json
import requests
from collections import Counter
from tqdm import tqdm

def get_majority_answer(answers):
    """Get the most common answer from the list."""
    return Counter(answers).most_common(1)[0][0]

def download(cache):
    """Download and process the TextVQA dataset."""
    dataset_path = os.path.join(cache, "text_vqa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    # Download annotations
    print("Downloading TextVQA annotations...")
    json_url = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json"
    json_path = os.path.join(dataset_path, "TextVQA_train.json")
    
    if not os.path.exists(json_path):
        response = requests.get(json_url)
        with open(json_path, 'w') as f:
            f.write(response.text)
    
    # Process annotations
    print("Processing annotations...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed_data = {}
    
    for item in tqdm(data['data']):
        image_path = f"text_vqa/train_images/{item['image_id']}.jpg"
        
        # Create caption from image classes if available
        caption = f"This image contains: {', '.join(item['image_classes'])}" if item.get('image_classes') else ""
        
        # Get majority answer
        answer = get_majority_answer(item['answers'])
        
        if image_path not in processed_data:
            processed_data[image_path] = {
                'image_id': item['image_id'],
                'captions': [caption] if caption else [],
                'QAs': []
            }
        
        processed_data[image_path]['QAs'].append({
            'question': item['question'],
            'answer': answer,
            'question_id': item['question_id']
        })
    
    # Save processed data
    processed_json_path = os.path.join(dataset_path, "processed_qa.json")
    with open(processed_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} images with QA pairs")
    
    # Create processed flag
    with open(processed_flag, 'w') as f:
        f.write("")

def load(cache):
    """Load the processed dataset."""
    dataset_path = os.path.join(cache, "text_vqa")
    processed_json_path = os.path.join(dataset_path, "processed_qa.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    text_vqa_data = {}
    
    for image_path, data in processed_data.items():
        text_vqa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "text_vqa",
            "captions": data['captions'],
            "bboxes": [],  # No bounding boxes in this dataset
            "QA": [ qa['question'] + " " + qa['answer'] for qa in data['QAs'] ]
        }
    
    return text_vqa_data

def info():
    """Return information about the TextVQA dataset."""
    return {
        "name": "TextVQA",
        "description": "A dataset for visual question answering about text in images",
        "size": "~10MB",
        "num_datapoints": 56555,
        "num_images": 21953,
        "publication_year": 2019,
        "requires_gpt_conversion": False
    }