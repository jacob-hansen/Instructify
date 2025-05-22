import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import io

def normalize_bbox(bbox, width, height):
    """Normalize bounding box coordinates to 0-1 range."""
    x1, y1, x2, y2 = bbox
    return [
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height
    ]

def process_qa_pairs(questions, answers):
    """Process questions and answers into formatted QA pairs."""
    qa_pairs = []
    for q_dict in questions:
        if isinstance(q_dict, dict) and 'question' in q_dict and 'answers' in q_dict:
            qa_pairs.append(f"Question: {q_dict['question']} Answer: {', '.join(q_dict['answers'])}")
    return qa_pairs

def download(cache):
    """Process the ST-VQA dataset."""
    dataset_path = os.path.join(cache, "st_vqa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading ST-VQA dataset...")
    dataset = load_dataset("vikhyatk/st-vqa")
    
    # Prepare directories
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Process data and save images
    processed_data = {}
    errors = []
    
    print("Processing data and saving images...")
    for split in ['train']:
        for idx, example in tqdm(enumerate(dataset[split])):
            try:
                # Generate unique image ID
                image_id = f"stvqa_{idx}"
                
                # Save image
                image_path = os.path.join(images_dir, f"{image_id}.jpg")
                if not os.path.exists(image_path):
                    image = example['image']
                    image.save(image_path, quality=95)
                
                # Process QA pairs
                qas = []
                if isinstance(example['qas'], list):
                    for qa in example['qas']:
                        if isinstance(qa, dict):
                            question = qa.get('question', '')
                            answers = qa.get('answers', [])
                            if isinstance(answers, list):
                                qas.append(f"Question: {question} Answer: {', '.join(answers)}")
                
                # Store processed data
                processed_data[f"st_vqa/images/{image_id}.jpg"] = {
                    'image_id': image_id,
                    'captions': [],  # ST-VQA doesn't have captions
                    'bboxes': [],    # Add OCR bbox processing if available in the dataset
                    'qa_pairs': qas
                }
                
            except Exception as e:
                errors.append((image_id, str(e)))
                if len(errors) < 5:
                    print(f"Error processing {image_id}: {str(e)}")
                continue
    
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
    dataset_path = os.path.join(cache, "st_vqa")
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    st_vqa_data = {}
    
    for image_path, data in processed_data.items():
        st_vqa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "st_vqa",
            "captions": data['captions'],
            "bboxes": data['bboxes'],
            "QA": data['qa_pairs']
        }
    
    return st_vqa_data

def info():
    """Return information about the ST-VQA dataset."""
    return {
        "name": "ST-VQA",
        "description": "Scene Text Visual Question Answering dataset",
        "size": "3 GB",
        "num_datapoints": 26070,
        "num_images": 18917,
        "publication_year": 2019,
        "requires_gpt_conversion": False
    }