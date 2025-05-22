import os
import json

def download(cache):
    """
    Since the DocOwl OCR dataset needs to be manually downloaded, this function
    simply checks if the data exists and raises an error if it doesn't.
    
    Args:
        cache (str): Path to the cache directory
    
    Raises:
        FileNotFoundError: If the dataset is not found in the expected location
    """
    dataset_path = os.path.join(cache, "docowl", "ocr_results.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"DocOwl dataset file not found: ocr_results.json. "
            f"This file must be manually downloaded to: {os.path.join(cache, 'docowl/')}"
        )
    print("Required dataset file found at expected location.")

def load(cache):
    """
    Loads the DocOwl OCR dataset and formats it into the expected structure.
    
    Args:
        cache (str): Path to the cache directory
    
    Returns:
        dict: Dictionary mapping image paths to their annotations
    """
    dataset_path = os.path.join(cache, "docowl", "ocr_results.json")
    
    # Read the JSON file
    with open(dataset_path, "r") as f:
        docowl_ocr = json.load(f)
    
    # Convert to the expected format
    formatted_data = {}
    
    for image_path, ocr_results in docowl_ocr.items():
        # Extract image_id from the path
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        
        formatted_data[image_path] = {
            "image_id": image_id,
            "image_source": "docowl",
            "bboxes": [],
            "captions": [],  # No caption data available
            "QA": []        # No QA data available
        }
        
        # Convert OCR results to bounding boxes
        # Each OCR result is [text, x1, y1, x2, y2]
        for ocr_result in ocr_results:
            formatted_data[image_path]["bboxes"].append([
                ocr_result[0],  # OCR text
                ocr_result[1],  # x1
                ocr_result[2],  # y1
                ocr_result[3],  # x2
                ocr_result[4]   # y2
            ])
    
    return formatted_data

def info():
    """
    Returns information about the DocOwl OCR dataset.
    
    Returns:
        dict: Dataset information
    """
    return {
        "name": "DocOwl OCR",
        "description": "OCR results for the DocOwl dataset containing text detection and recognition.",
        "size": "Varies",
        "num_datapoints": 826192,
        "num_images": 55056,
        "publication_year": 2023,
        "requires_gpt_conversion": True # it did require it before
    }