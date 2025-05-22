import os
import json
import numpy as np
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

def download(cache):
    """Process the DocVQA dataset."""
    dataset_path = os.path.join(cache, "doc_vqa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading DocVQA dataset...")
    dataset = load_dataset("cmarkea/doc-vqa")
    
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
                # Use the provided ID as image_id
                image_id = example['id']
                
                # Save image
                image_path = os.path.join(images_dir, f"{image_id}.jpg")
                if not os.path.exists(image_path):
                    image = example['image']
                    image.save(image_path, quality=95)
                
                # Process QA pairs
                qa_pairs = []
                if 'qa' in example and isinstance(example['qa'], dict):
                    qa_data = example['qa']
                    if 'en' in qa_data and isinstance(qa_data['en'], list):
                        for qa_item in qa_data['en']:
                            if isinstance(qa_item, dict):
                                question = qa_item.get('question', '')
                                answer = qa_item.get('answer', '')
                                if question and answer:
                                    qa_pairs.append(f"Question: {question} Answer: {answer}")
                
                # Store metadata
                metadata = {
                    'paper_id': example.get('paper_id', ''),
                    'source': example.get('source', '')
                }
                
                # Store processed data
                processed_data[f"doc_vqa/images/{image_id}.jpg"] = {
                    'image_id': image_id,
                    'captions': [],  # DocVQA doesn't have captions
                    'bboxes': [],    # Add OCR bbox processing if available
                    'qa_pairs': qa_pairs,
                    'metadata': metadata
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
    dataset_path = os.path.join(cache, "doc_vqa")
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    doc_vqa_data = {}
    
    for image_path, data in processed_data.items():
        doc_vqa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "doc_vqa",
            "captions": data['captions'],
            "bboxes": data['bboxes'],
            "QA": data['qa_pairs'],
            "metadata": data.get('metadata', {})
        }
    
    return doc_vqa_data

def info():
    """Return information about the DocVQA dataset."""
    return {
        "name": "DocVQA",
        "description": "Document Visual Question Answering dataset",
        "size": "??",
        "num_datapoints": 9690,  # Based on the preview showing 9.69k rows
        "num_images": 9690,      # Assuming one image per datapoint
        "publication_year": 2021, # Update if different
        "requires_gpt_conversion": False
    }