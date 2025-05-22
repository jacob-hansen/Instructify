import os
import json

def download(cache):
    """
    Since the LLaVaR OCR dataset needs to be manually downloaded, this function
    simply checks if the data exists and raises an error if it doesn't.
    
    Args:
        cache (str): Path to the cache directory
    
    Raises:
        FileNotFoundError: If the dataset is not found in the expected location
    """
    dataset_path = os.path.join(cache, "llavar", "llavar_ocr_results.jsonl")
    metadata_path = os.path.join(cache, "llavar", "finetune_meta.json")
    
    missing_files = []
    if not os.path.exists(dataset_path):
        missing_files.append("llavar_ocr_results.jsonl")
    if not os.path.exists(metadata_path):
        missing_files.append("finetune_meta.json")
        
    if missing_files:
        raise FileNotFoundError(
            f"LLaVaR dataset files not found: {', '.join(missing_files)}. "
            f"These files must be manually downloaded to: {os.path.join(cache, 'llavar/')}"
        )
    print("All required dataset files found at expected location.")

def load(cache):
    """
    Loads the LLaVaR OCR dataset and metadata, combining them into the expected structure.
    
    Args:
        cache (str): Path to the cache directory
    
    Returns:
        dict: Dictionary mapping image paths to their annotations
    """
    dataset_path = os.path.join(cache, "llavar", "llavar_ocr_results.jsonl")
    metadata_path = os.path.join(cache, "llavar", "finetune_meta.json")
    
    # Read the metadata file
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Create a mapping of llavar_id to metadata
    metadata_map = {item['llavar_id']: item for item in metadata}
    
    # Read the JSONL file
    with open(dataset_path, "r") as f:
        llavar_ocr = [json.loads(line) for line in f.readlines()]
    
    # Convert to the expected format
    formatted_data = {}
    
    for entry in llavar_ocr:
        image_id = entry['llavar_id']
        image_path = f"llavar/llavar_images/{image_id}"
        
        # Get corresponding metadata
        meta = metadata_map.get(image_id, {})
        
        formatted_data[image_path] = {
            "image_id": image_id,
            "image_source": "llavar",
            "bboxes": [],
            "captions": [meta.get('TEXT', '')] if meta.get('TEXT') else [],  # Add TEXT as caption
            "QA": []  # No QA data
        }
        
        # Add image dimensions if available
        if 'WIDTH' in meta and 'HEIGHT' in meta:
            formatted_data[image_path]["width"] = meta['WIDTH']
            formatted_data[image_path]["height"] = meta['HEIGHT']
        
        # Convert OCR results to bounding boxes
        # Each OCR result is [text, x1, y1, x2, y2]
        for ocr_result in entry['ocr_results']:
            formatted_data[image_path]["bboxes"].append([
                ocr_result[0],  # OCR text as the "category"
                ocr_result[1],  # x1
                ocr_result[2],  # y1
                ocr_result[3],  # x2
                ocr_result[4]   # y2
            ])
    
    return formatted_data

def info():
    """
    Returns information about the LLaVaR OCR dataset.
    
    Returns:
        dict: Dataset information
    """
    return {
        "name": "LLaVaR OCR",
        "description": "OCR results for the LLaVaR dataset containing text detection and recognition.",
        "size": "Varies",  # Add actual size if known
        "num_datapoints": None,  # Add if known
        "num_images": None,      # Add if known
        "publication_year": None, # Add if known
        "requires_gpt_conversion": False
    }