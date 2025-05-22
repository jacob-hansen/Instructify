import os
import json
from datasets import load_dataset
from tqdm import tqdm
import io
import hashlib
from PIL import Image

def get_image_hash(image):
    """Generate a hash for an image to identify duplicates.
    
    Args:
        image: A PIL Image object
    Returns:
        str: A hex digest of the image hash
    """
    # Convert to RGB to ensure consistent format
    image = image.convert('RGB')
    
    # Convert image to bytes in a consistent format
    with io.BytesIO() as byte_io:
        image.save(byte_io, format='PNG')  # Use PNG for consistent encoding
        image_bytes = byte_io.getvalue()
    
    # Create hash
    return hashlib.sha256(image_bytes).hexdigest()[:16]  # Use first 16 chars for shorter IDs

def download(cache):
    """Process the ChartQA dataset."""
    dataset_path = os.path.join(cache, "chart_qa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    # if os.path.exists(processed_flag):
    #     print("Dataset already processed.")
    #     return
    
    print("Loading ChartQA dataset...")
    dataset = load_dataset("HuggingFaceM4/ChartQA")
    
    # Prepare directories
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Process data and save images
    processed_data = {}
    errors = []
    image_hashes = {}  # Map from image hash to image_id
    
    print("Processing data and saving images...")
    for split in ['train']:
        for idx, example in tqdm(enumerate(dataset[split])):
            try:
                # Generate hash for the image
                image = example['image']
                image_hash = get_image_hash(image)
                
                # Use existing image_id if we've seen this hash before
                if image_hash in image_hashes:
                    image_id = image_hashes[image_hash]
                else:
                    # Generate new image_id for new unique images
                    image_id = f"chartqa_{image_hash}"
                    image_hashes[image_hash] = image_id
                    
                    # Save image only if we haven't seen it before
                    image_path = os.path.join(images_dir, f"{image_id}.jpg")
                    if not os.path.exists(image_path):
                        image.convert("RGB").save(image_path, quality=95)
                
                # Process QA pair
                qa_pairs = [f"Question: {example['query']} Answer: {label}" for label in example['label']]
                
                # Store processed data
                image_path = f"chart_qa/images/{image_id}.jpg"
                if image_path not in processed_data:
                    processed_data[image_path] = {
                        'image_id': image_id,
                        'captions': [],  # ChartQA doesn't have captions
                        'bboxes': [],    # ChartQA doesn't have bounding boxes
                        'qa_pairs': qa_pairs
                    }
                else:
                    # Append new QA pairs to existing image data
                    processed_data[image_path]['qa_pairs'].extend(qa_pairs)
                
            except Exception as e:
                errors.append((str(idx), str(e)))
                if len(errors) < 5:
                    print(f"Error processing index {idx}: {str(e)}")
                continue
    
    # Save processed data
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    with open(processed_json_path, 'w') as f:
        json.dump(processed_data, f)
    
    # Save hash mapping for reference
    hash_mapping_path = os.path.join(dataset_path, "image_hash_mapping.json")
    with open(hash_mapping_path, 'w') as f:
        json.dump(image_hashes, f, indent=2)
    
    print(f"Processed {len(processed_data)} unique images")
    print(f"Total image hashes: {len(image_hashes)}")
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
    dataset_path = os.path.join(cache, "chart_qa")
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    chart_qa_data = {}
    
    for image_path, data in processed_data.items():
        chart_qa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "chart_qa",
            "captions": data['captions'],
            "bboxes": data['bboxes'],
            "QA": data['qa_pairs']
        }
    
    return chart_qa_data

def info():
    """Return information about the ChartQA dataset."""
    return {
        "name": "ChartQA",
        "description": "Chart Question Answering dataset with human and machine generated questions",
        "size": "0.9 GB",
        "num_datapoints": 28299,
        "num_images": 18271,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }