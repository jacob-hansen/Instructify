import os
import json
from PIL import Image
import numpy as np
from datasets import load_dataset
import io
from tqdm import tqdm
import pandas as pd

def merge_images(img1, img2):
    """Merge two images side by side."""
    
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
    """Process the ImageEditingRequestV1 dataset."""
    dataset_path = os.path.join(cache, "image_editing_request")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading ImageEditingRequestV1 dataset...")
    dataset = load_dataset("taesiri/ImageEditingRequestV1")
    
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
            try:
                # Create merged image filename
                merged_filename = f"{example['uid']}.jpg"
                merged_path = os.path.join(images_dir, merged_filename)
                
                # Merge images and save
                merged_image = merge_images(example['img0'], example['img1'])
                merged_image.save(merged_path, quality=95)
                
                # Format the editing instruction to reference left/right images
                instruction = f"The left image was edited to create the right image: {example['sents'][0]}"
                
                # Add metadata row
                metadata_rows.append({
                    'image_path': f"image_editing_request/images/{merged_filename}",
                    'instruction': instruction,
                    'uid': example['uid'],
                    'original_filename': example['img0_filename'],
                    'edited_filename': example['img1_filename']
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
    dataset_path = os.path.join(cache, "image_editing_request")
    csv_path = os.path.join(dataset_path, "metadata.csv")
    
    # Read metadata CSV
    df = pd.read_csv(csv_path)
    
    edit_request_data = {}
    
    for _, row in df.iterrows():
        image_path = row['image_path']
        
        edit_request_data[image_path] = {
            "image_id": row['uid'],
            "image_source": "image_editing_request",
            "captions": [row['instruction']],
            "bboxes": [],    # No bounding boxes in this dataset
            "QA": [],        # No QA pairs in this dataset
            "metadata": {
                "original_filename": row['original_filename'],
                "edited_filename": row['edited_filename']
            }
        }
    
    return edit_request_data

def info():
    """Return information about the ImageEditingRequestV1 dataset."""
    return {
        "name": "ImageEditingRequestV1",
        "description": "A dataset containing pairs of original and edited images with editing instructions",
        "size": "294 MB",
        "num_datapoints": 3061,
        "num_images": 3061,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }