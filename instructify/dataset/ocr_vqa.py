import os
import json
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

def process_ocr_info(ocr_info, width, height):
    """Convert OCR info into normalized bounding boxes with labels."""
    bboxes = []
    for item in ocr_info:
        if not isinstance(item, dict) or 'word' not in item or 'bounding_box' not in item:
            continue
            
        bbox = item['bounding_box']
        if not all(key in bbox for key in ['x', 'y', 'width', 'height']):
            continue
            
        # Convert x,y,width,height to x1,y1,x2,y2
        x1 = bbox['x']
        y1 = bbox['y']
        x2 = x1 + bbox['width']
        y2 = y1 + bbox['height']
        
        # Normalize coordinates
        normalized_bbox = normalize_bbox([x1, y1, x2, y2], width, height)
        
        bboxes.append([item['word']] + normalized_bbox)
    
    return bboxes

def download(cache):
    """Process the OCR-VQA dataset."""
    dataset_path = os.path.join(cache, "ocr_vqa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading OCR-VQA dataset...")
    dataset = load_dataset("howard-hou/OCR-VQA")
    
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
                # Save image
                image_path = os.path.join(images_dir, f"{example['image_id']}.jpg")
                if not os.path.exists(image_path):
                    image = example['image']
                    image.save(image_path, quality=95)
                
                # Format caption
                caption_parts = []
                if example.get('title'):
                    caption_parts.append(f"Title: {example['title']}")
                if example.get('authorName'):
                    caption_parts.append(f"Author: {example['authorName']}")
                if example.get('genre'):
                    caption_parts.append(f"Genre: {example['genre']}")
                
                # Process OCR info and QA pairs
                bboxes = process_ocr_info(example['ocr_info'], example['image_width'], example['image_height'])
                
                # Pair up questions and answers
                qa_pairs = []
                for q, a in zip(example['questions'], example['answers']):
                    qa_pairs.append("Question: " + q + " Answer: " + a)
                
                # Store processed data
                processed_data[f"ocr_vqa/images/{example['image_id']}.jpg"] = {
                    'image_id': example['image_id'],
                    'captions': caption_parts,
                    'bboxes': bboxes,
                    'qa_pairs': qa_pairs
                }
                
            except Exception as e:
                errors.append((example['image_id'], str(e)))
                if len(errors) < 5:
                    print(f"Error processing {example['image_id']}: {str(e)}")
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
    dataset_path = os.path.join(cache, "ocr_vqa")
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    ocr_vqa_data = {}
    
    for image_path, data in processed_data.items():
        ocr_vqa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "ocr_vqa",
            "captions": data['captions'],
            "bboxes": data['bboxes'],
            "QA": data['qa_pairs']
        }
    
    return ocr_vqa_data

def info():
    """Return information about the OCR-VQA dataset."""
    return {
        "name": "OCR-VQA",
        "description": "A dataset for visual question answering about text in images with OCR annotations",
        "size": "??",
        "num_datapoints": 967512,
        "num_images": 166007,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }