import os
import json
import io
from datasets import load_dataset
import pandas as pd

def download(cache):
    """
    Load the dataset using Hugging Face's datasets library and save to local CSV.
    Saves images and creates a CSV file containing all metadata.
    """
    dataset_path = os.path.join(cache, "dior_rsvg")
    os.makedirs(dataset_path, exist_ok=True)
    
    # Check if the dataset is already processed
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return

    # Load dataset using Hugging Face's datasets library
    print("Loading DIOR-RSVG dataset...")
    dataset = load_dataset("danielz01/DIOR-RSVG")
    
    # Save images and collect metadata
    print("Processing and saving images...")
    images_dir = os.path.join(dataset_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    # Process train split
    for idx, example in enumerate(dataset['train']):
        # Save image
        image = example['image']
        image_path = os.path.join(images_dir, f"{example['path']}")
        if not os.path.exists(image_path):
            image.save(image_path)
        
        # Get image dimensions for normalization
        width, height = image.size
        
        # Parse objects
        if isinstance(example['objects'], str):
            objects = json.loads(example['objects'])
        else:
            objects = example['objects']
        
        # Process bounding boxes and create normalized coordinates
        bboxes = []
        for bbox, category, caption in zip(objects["bbox"], objects["categories_normalized"], objects.get("captions", [])):
            # Assuming original bbox format is [x1, y1, x2, y2, label]
            # Normalize coordinates
            x1 = bbox[0] / width
            y1 = bbox[1] / height
            x2 = bbox[2] / width
            y2 = bbox[3] / height
            bboxes.append([category, x1, y1, x2, y2])
            if caption:
                bboxes.append([caption, x1, y1, x2, y2])
        
        # Add to CSV data
        csv_data.append({
            'image_path': f"dior_rsvg/images/{example['path']}",
            'image_id': example['path'],
            'bboxes': json.dumps(bboxes),
        })
        
        if idx % 1000 == 0:
            print(f"Processed {idx} images...")
    
    # Save metadata to CSV
    csv_path = os.path.join(dataset_path, "metadata.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    # Create a flag to indicate successful processing
    with open(processed_flag, 'w') as f:
        f.write("")

def load(cache):
    """
    Load the processed dataset from the cache directory using the CSV file.
    Returns a dictionary containing the dataset information.
    """
    dataset_path = os.path.join(cache, "dior_rsvg")
    csv_path = os.path.join(dataset_path, "metadata.csv")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    dior_data = {}
    
    # Process each row in the CSV
    for _, row in df.iterrows():
        dior_data[row['image_path']] = {
            "image_id": row['image_id'],
            "image_source": "dior_rsvg",
            "bboxes": json.loads(row['bboxes']),
            "captions": [],  # No captions in this dataset
            "QA": [],  # No QA pairs in this dataset
        }
    
    return dior_data

def info():
    """
    Return information about the DIOR-RSVG dataset.
    """
    return {
        "name": "DIOR-RSVG",
        "description": "A dataset containing aerial/satellite imagery with object detection annotations and captions",
        "size": "7.9GB",
        "num_datapoints": 53982,
        "num_images": 14748,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }