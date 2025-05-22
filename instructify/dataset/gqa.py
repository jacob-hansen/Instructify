import os
import json
import requests
import tqdm
from zipfile import ZipFile

# Images download:
# https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

def download(cache):
    """
    Download GQA dataset files and extract them to the specified cache directory.
    
    Args:
        cache (str): Path to the cache directory
    """
    # Directory where the dataset files will be stored
    dataset_dir = os.path.join(cache, "gqa")
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_dir, "downloaded")
    if os.path.exists(downloaded_flag):
        print("Dataset already downloaded.")
        return

    # URLs for the dataset files
    urls = [
        "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip",
        "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
    ]

    # Download and unzip each file
    for url in urls:
        file_name = os.path.join(dataset_dir, os.path.basename(url))
        if not os.path.exists(file_name.replace('.zip', '')):
            print(f"Downloading {url} to {file_name}...")
            response = requests.get(url)
            with open(file_name, "wb") as f:
                f.write(response.content)
            with ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            os.remove(file_name)

    # Create a file to indicate the dataset has been downloaded
    with open(os.path.join(dataset_dir, "downloaded"), "w") as f:
        f.write("")

def process_scene_graph(scene_data):
    """
    Process a single scene graph into a standardized format.
    
    Args:
        scene_data (dict): Raw scene graph data for one image
        
    Returns:
        list: List of processed bounding boxes with labels and attributes
    """
    bboxes_formatted = []
    
    # First count occurrences of each object name
    object_counts = {}
    for obj_info in scene_data['objects'].values():
        obj_name = obj_info['name']
        object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
    
    for obj_id, obj_info in scene_data['objects'].items():
        # Normalize bbox coordinates
        x = obj_info['x'] / scene_data['width']
        y = obj_info['y'] / scene_data['height']
        w = obj_info['w'] / scene_data['width']
        h = obj_info['h'] / scene_data['height']
        bbox = [x, y, x + w, y + h]
        
        # Add main object with its name
        bboxes_formatted.append([obj_info['name']] + bbox)
        
        # Add attributes
        if 'attributes' in obj_info:
            for attr in obj_info['attributes']:
                bboxes_formatted.append([attr] + bbox)
        
        # Add relations
        if 'relations' in obj_info:
            for relation in obj_info['relations']:
                # Get the target object's name
                target_obj_id = relation['object']
                if target_obj_id in scene_data['objects']:
                    target_obj_name = scene_data['objects'][target_obj_id]['name']
                    # Choose article based on count
                    article = 'a' if object_counts[target_obj_name] > 1 else 'the'
                    # Handle special cases for articles
                    if article == 'a' and target_obj_name[0].lower() in 'aeiou':
                        article = 'an'
                    relation_str = f"{relation['name']} {article} {target_obj_name}"
                    bboxes_formatted.append([relation_str] + bbox)
    
    return bboxes_formatted

def process_question(question_data, image_id):
    """
    Process a single question into a standardized format.
    
    Args:
        question_data (dict): Raw question data
        image_id (str): ID of the image the question refers to
        
    Returns:
        dict: Processed question data
    """
    return question_data['question'] + " " + question_data['fullAnswer']

def load(cache):
    """
    Load the GQA dataset from the cache directory.
    
    Args:
        cache (str): Path to the cache directory
        
    Returns:
        dict: Processed GQA dataset
    """
    dataset_dir = os.path.join(cache, "gqa")
    
    # Check if the dataset has been downloaded
    if not os.path.exists(os.path.join(dataset_dir, "downloaded")):
        raise FileNotFoundError("GQA dataset not found. Please download it first using the download() function.")

    # Load scene graphs
    scene_graphs = {}
    scene_graph_files = ['train_sceneGraphs.json', 'val_sceneGraphs.json']
    
    for sg_file in scene_graph_files:
        sg_path = os.path.join(dataset_dir, sg_file)
        if os.path.exists(sg_path):
            with open(sg_path, 'r') as f:
                scene_graphs.update(json.load(f))

    # Load questions and create image-to-questions mapping
    image_to_questions = {}
    q_path = os.path.join(dataset_dir, 'train_balanced_questions.json')
    if os.path.exists(q_path):
        with open(q_path, 'r') as f:
            questions = json.load(f)
            for q_id, q_data in questions.items():
                img_id = q_data['imageId']
                if img_id not in image_to_questions:
                    image_to_questions[img_id] = []
                image_to_questions[img_id].append(q_data)

    # Process the dataset
    gqa_dataset = {}
    
    # Process each image in the scene graphs
    for image_id, scene_data in scene_graphs.items():
        # Get and process questions for this image
        image_questions = [
            process_question(q_data, image_id) 
            for q_data in image_to_questions.get(image_id, [])
        ]
        
        # Format the entry
        gqa_dataset[f"gqa/images/{image_id}.jpg"] = {
            "image_id": image_id,
            "image_source": "gqa",
            "bboxes": process_scene_graph(scene_data),
            "captions": [],  # GQA doesn't have captions
            "QA": image_questions
        }
    
    return gqa_dataset

def info():
    """
    Return information about the GQA dataset.
    
    Returns:
        dict: Dataset information
    """
    return {
        "name": "GQA",
        "description": "GQA is a Visual Question Answering dataset with scene graph annotations and compositional questions.",
        "size": "3000 MB",
        "num_datapoints": 22000000,
        "num_images": 113000,
        "publication_year": 2019,
        "requires_gpt_conversion": False
    }