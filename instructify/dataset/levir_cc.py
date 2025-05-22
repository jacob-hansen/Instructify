import os
import json
import gdown
import zipfile
from PIL import Image
from tqdm import tqdm
import pandas as pd

def merge_images(img1_path, img2_path):
    """Merge two images side by side."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
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
    """Download and process the Levir-CC dataset."""
    dataset_path = os.path.join(cache, "levir_cc")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    # Download dataset
    print("Downloading Levir-CC dataset...")
    zip_path = os.path.join(dataset_path, "levir_cc.zip")
    if not os.path.exists(zip_path):
        url = "https://drive.google.com/uc?id=1YgWES-0OOL-3KK4yIV3-y8MN-_c1ELEE"
        gdown.download(url, zip_path, quiet=False)
    
    # Extract dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    
    # Clean up zip file
    os.remove(zip_path)
    
    # Load JSON annotations
    print("Processing annotations...")
    json_path = os.path.join(dataset_path, "LevirCCcaptions.json")
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    # Prepare directories
    merged_images_dir = os.path.join(dataset_path, "merged_images")
    if not os.path.exists(merged_images_dir):
        os.makedirs(merged_images_dir)
    
    # Process images and prepare metadata
    metadata_rows = []
    errors = []
    
    print("Processing and merging images...")
    for image_info in tqdm(annotations['images']):
        try:
            # Get paths for both images
            base_filename = image_info['filename']
            img_a_path = os.path.join(dataset_path, "images", 
                                    image_info['split'], "A", base_filename)
            img_b_path = os.path.join(dataset_path, "images", 
                                    image_info['split'], "B", base_filename)
            
            # Create merged image filename
            merged_filename = f"merged_{image_info['split']}_{base_filename}"
            merged_path = os.path.join(merged_images_dir, merged_filename)
            
            # Merge and save images
            merged_image = merge_images(img_a_path, img_b_path)
            merged_image.save(merged_path, quality=95)
            
            # Get all captions
            captions = [sent['raw'].strip() for sent in image_info['sentences']]
            
            # Add metadata row
            metadata_rows.append({
                'image_path': f"levir_cc/merged_images/{merged_filename}",
                'captions': json.dumps(captions),
                'split': image_info['split'],
                'change_flag': image_info['changeflag'],
                'image_id': image_info['imgid']
            })
            
        except Exception as e:
            errors.append((base_filename, str(e)))
            if len(errors) < 5:
                print(f"Error processing {base_filename}: {str(e)}")
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
    dataset_path = os.path.join(cache, "levir_cc")
    csv_path = os.path.join(dataset_path, "metadata.csv")
    
    # Read metadata CSV
    df = pd.read_csv(csv_path)
    
    levir_cc_data = {}
    
    for _, row in df.iterrows():
        image_path = row['image_path']
        
        levir_cc_data[image_path] = {
            "image_id": str(row['image_id']),
            "image_source": "levir_cc",
            "captions": json.loads(row['captions']),
            "bboxes": [],    # No bounding boxes in this dataset
            "QA": [],        # No QA pairs in this dataset
            "metadata": {
                "split": row['split'],
                "change_flag": row['change_flag']
            }
        }
    
    return levir_cc_data

def info():
    """Return information about the Levir-CC dataset."""
    return {
        "name": "Levir-CC",
        "description": "A dataset of satellite image pairs with change captions",
        "size": "3.1 GB",
        "num_datapoints": 50385,
        "num_images": 10077,
        "publication_year": 2023,
        "requires_gpt_conversion": False
    }