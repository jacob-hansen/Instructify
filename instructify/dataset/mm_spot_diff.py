import os
import csv
import json
from PIL import Image
import numpy as np
from datasets import load_dataset
import io
from tqdm import tqdm
import pandas as pd

def merge_images(img1_dict, img2_dict):
    """Merge two images side by side."""
    # Convert bytes directly to PIL Images - no need for base64 decoding
    img1 = Image.open(io.BytesIO(img1_dict['bytes'])).convert("RGB")
    img2 = Image.open(io.BytesIO(img2_dict['bytes'])).convert("RGB")
    
    # Convert to RGB mode if needed
    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')
    
    # Ensure both images have the same height
    max_height = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * max_height / img1.height), max_height))
    img2 = img2.resize((int(img2.width * max_height / img2.height), max_height))
    
    # Create new image with combined width
    merged_width = img1.width + img2.width
    merged_image = Image.new('RGB', (merged_width, max_height))
    
    # Paste images side by side
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (img1.width, 0))
    
    return merged_image

def download(cache):
    """Download and process the MM Spot the Diff dataset."""
    dataset_path = os.path.join(cache, "mm_spot_diff")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading MM Spot the Diff dataset...")
    dataset = load_dataset("nimapourjafar/mm_spot_the_diff")
    
    # Prepare directories
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    # Prepare metadata for CSV
    metadata_rows = []
    errors = []
    
    print("Processing and merging images...")
    for split in ['train']:
        for idx, example in tqdm(enumerate(dataset[split])):
            # Create merged image filename
            merged_filename = f"merged_{split}_{idx:06d}.jpg"
            merged_path = os.path.join(images_dir, merged_filename)
            
            # Get images from the dataset
            images = example['images']
            if len(images) >= 2:  # Ensure we have both images
                try:
                    # Merge images and save
                    merged_image = merge_images(images[0], images[1])
                    merged_image.save(merged_path, quality=95)
                    
                    # Extract conversation data
                    conversation = []
                    for item in example['data']:
                        if isinstance(item, dict):
                            role = item.get('role', '')
                            data = item.get('data', '')
                            modality = item.get('modality', '')
                            conversation.append({
                                'role': role,
                                'data': data,
                                'modality': modality
                            })
                    
                    # Add metadata row
                    metadata_rows.append({
                        'image_path': f"mm_spot_diff/images/{merged_filename}",
                        'conversation': json.dumps(conversation),
                        'split': split
                    })
                except Exception as e:
                    errors.append((idx, str(e)))
                    if len(errors) < 5:  # Print first few errors for debugging
                        print(f"Error processing image pair {idx}: {str(e)}")
                    continue
    
    # Save metadata to CSV
    csv_path = os.path.join(dataset_path, "metadata.csv")
    df = pd.DataFrame(metadata_rows)
    df.to_csv(csv_path, index=False)
    
    print(f"Processed {len(metadata_rows)} image pairs successfully")
    print(f"Failed to process {len(errors)} pairs")
    
    # Save error log
    if errors:
        error_path = os.path.join(dataset_path, "processing_errors.json")
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
    
    # Create processed flag
    with open(processed_flag, 'w') as f:
        f.write("")

def load(cache):
    """Load the processed dataset."""
    dataset_path = os.path.join(cache, "mm_spot_diff")
    csv_path = os.path.join(dataset_path, "metadata.csv")
    
    # Read metadata CSV
    df = pd.read_csv(csv_path)
    
    spot_diff_data = {}
    
    for _, row in df.iterrows():
        image_path = row['image_path']
        conversation = json.loads(row['conversation'])
        
        # Extract question and answer from conversation
        question = next((item['data'] for item in conversation 
                        if item['role'] == 'user' and item['modality'] == 'text'), 
                       "What are the differences between these images?")
        
        answer = next((item['data'] for item in conversation 
                      if item['role'] == 'assistant' and item['modality'] == 'text'),
                     "")
        
        spot_diff_data[image_path] = {
            "image_id": os.path.basename(image_path),
            "image_source": "mm_spot_diff",
            "captions": [],  # No captions in this dataset
            "bboxes": [],    # No bounding boxes in this dataset
            "QA": "Question: " + question + " Answer: " + answer,
            "split": row['split']
        }
    
    return spot_diff_data

def info():
    """Return information about the MM Spot the Diff dataset."""
    return {
        "name": "MM Spot the Diff",
        "description": "A dataset containing pairs of similar images with descriptions of their differences",
        "size": "1.55 GB",
        "num_datapoints": 8566,
        "num_images": 8566,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }