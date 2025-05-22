import os
import json
import io
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

def download(cache):
    """Process the MM VisualMRC dataset, focusing only on images."""
    dataset_path = os.path.join(cache, "mm_visualmrc")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading MM VisualMRC dataset...")
    dataset = load_dataset("nimapourjafar/mm_visualmrc")
    
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
            try:
                # Generate a unique image ID
                image_id = f"{split}_{idx}"
                
                # Process only if there's an image
                if example['images'] and len(example['images']) > 0:
                    # Save image
                    image_path = os.path.join(images_dir, f"{image_id}.jpg")
                    if not os.path.exists(image_path):
                        assert len(example['images']) == 1, "Expected exactly one image per example"
                        if not os.path.exists(image_path):
                            image = Image.open(io.BytesIO(example['images'][0]['bytes'])).convert("RGB")
                            image.save(image_path, quality=95)
                    
                    # Convert conversation to QA pairs
                    qa_pairs = []
                    data_entries = [i for i in example.get('data', []) if i.get("modality", "") != "image"]
                    
                    # Process entries in pairs to create QA pairs
                    for i in range(0, len(data_entries)-1, 2):
                        user_entry = data_entries[i]
                        assistant_entry = data_entries[i+1] if i+1 < len(data_entries) else None
                        
                        question = user_entry.get('data', '').strip()
                        answer = assistant_entry.get('data', '').strip()
                        
                        if question and answer:
                            qa_pair = f"Question: {question} Answer: {answer}"
                            qa_pairs.append(qa_pair)

                    # Store processed data
                    processed_data[f"mm_visualmrc/images/{image_id}.jpg"] = {
                        'image_id': image_id,
                        'captions': [],  # No captions in this dataset
                        'bboxes': [],    # No bboxes in this dataset
                        'qa_pairs': qa_pairs,
                        'metadata': {
                            'split': split
                        }
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
    dataset_path = os.path.join(cache, "mm_visualmrc")
    processed_json_path = os.path.join(dataset_path, "processed_data.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    mm_visualmrc_data = {}
    
    for image_path, data in processed_data.items():
        mm_visualmrc_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "mm_visualmrc",
            "captions": data['captions'],
            "bboxes": data['bboxes'],
            "QA": data['qa_pairs'],
            "metadata": data.get('metadata', {})
        }
    
    return mm_visualmrc_data

def info():
    """Return information about the MM VisualMRC dataset."""
    return {
        "name": "MM VisualMRC",
        "description": "Multimodal Visual Machine Reading Comprehension dataset",
        "size": "2.91 GB",
        "num_datapoints": 11988,
        "num_images": 3027,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }