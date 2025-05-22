import os
import json
import requests
import tarfile
from typing import Dict, Any

def download(cache: str) -> None:
    """
    Downloads the Visual News dataset and extracts it to the specified cache directory.
    
    Args:
        cache (str): Path to the cache directory where the dataset will be stored
    """
    dataset_path = os.path.join(cache, "visual_news")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # URLs for the Visual News dataset
    urls = {
        "articles": "https://www.cs.rice.edu/~vo9/visualnews/articles.tar.gz",
        "origin": "https://www.cs.rice.edu/~vo9/visualnews/origin.tar"
    }
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # Download and extract each file
    for name, url in urls.items():
        print(f"Downloading {name} dataset...")
        response = requests.get(url, stream=True)
        file_path = os.path.join(dataset_path, os.path.basename(url))
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Extracting {name} dataset...")
        if url.endswith('.tar.gz'):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(dataset_path)
        else:
            with tarfile.open(file_path, 'r') as tar:
                tar.extractall(dataset_path)
        
        # Remove the downloaded archive
        os.remove(file_path)
    
    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

def load(cache: str) -> Dict[str, Any]:
    """
    Loads the Visual News dataset and converts it to the COCO captions format.
    
    Args:
        cache (str): Path to the cache directory where the dataset is stored
        
    Returns:
        Dict[str, Any]: Dictionary containing the dataset in COCO captions format
    """
    dataset_path = os.path.join(cache, "visual_news")
    data_file = os.path.join(dataset_path, "origin", "data.json")
    
    # Load the original data
    with open(data_file, 'r', encoding='utf-8') as f:
        visual_news_data = json.load(f)
    
    # Convert to COCO captions format
    visual_news_captions = {}
    
    for item in visual_news_data:
        # Create a standardized image path
        image_path = f"visual_news/{item['source']}/images/{item['image_path'].split('/')[-2]}/{item['image_path'].split('/')[-1]}"
        
        # Initialize the entry if it doesn't exist
        if image_path not in visual_news_captions:
            visual_news_captions[image_path] = {
                "image_id": str(item['id']),
                "image_source": "visual_news",
                "bboxes": [],  # Visual News doesn't include bounding boxes
                "captions": [],
                "QA": [],     # Visual News doesn't include QA pairs
                "metadata": {  # Additional Visual News specific metadata
                    "topic": item['topic'],
                    "source": item['source'],
                    "article_path": item['article_path']
                }
            }
        
        # Add the caption
        if item['caption']:
            visual_news_captions[image_path]["captions"].append(item['caption'])
    
    return visual_news_captions

def info() -> Dict[str, Any]:
    """
    Returns information about the Visual News dataset.
    
    Returns:
        Dict[str, Any]: Dictionary containing dataset information
    """
    return {
        "name": "Visual News",
        "description": "A large-scale news image dataset with captions from major news agencies including USA Today, The Guardian, BBC News, and The Washington Post.",
        "size": "~2.5 GB",
        "num_datapoints": 1000000,  # Approximate
        "publication_year": 2020,
        "requires_gpt_conversion": False,
        "sources": ["usa_today", "guardian", "bbc", "washington_post"]
    }