import os
import json
import gdown
import zipfile
from tqdm import tqdm

def format_qa_string(problem):
    """Format QA into a clear string format."""
    # Get the correct answer choice
    correct_choice = problem['choices'][problem['answer']]
    
    # Format the question and answer
    qa_string = f"Question: {problem['question']}\n"
    
    # Add choices if present
    if problem['choices']:
        qa_string += "Choices:\n"
        for i, choice in enumerate(problem['choices']):
            qa_string += f"({chr(65+i)}) {choice}\n"
    
    qa_string += f"Answer: {correct_choice}\n"
    
    # Add solution if present
    if problem.get('solution'):
        qa_string += f"Solution: {problem['solution']}"
    
    return qa_string

def format_caption(problem):
    """Format educational context as caption."""
    caption_parts = []
    
    if problem.get('lecture'):
        caption_parts.append(f"Background: {problem['lecture']}")
    
    context_parts = []
    if problem.get('subject'):
        context_parts.append(f"Subject: {problem['subject']}")
    if problem.get('category'):
        context_parts.append(f"Category: {problem['category']}")
    if problem.get('skill'):
        context_parts.append(f"Skill: {problem['skill']}")
    if problem.get('topic'):
        context_parts.append(f"Topic: {problem['topic']}")
        
    if context_parts:
        caption_parts.append(" | ".join(context_parts))
        
    return " ".join(caption_parts)

def download(cache):
    """Download and process the ScienceQA dataset."""
    dataset_path = os.path.join(cache, "scienceqa")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    # Download images
    print("Downloading ScienceQA images...")
    images_zip_path = os.path.join(dataset_path, "images.zip")
    if not os.path.exists(images_zip_path):
        image_url = "https://drive.google.com/uc?id=1-enzrA8L3CvSY4_0hJjW4QgKlkirqiCW"
        gdown.download(image_url, images_zip_path, quiet=False)
    
    # Download problems
    print("Downloading ScienceQA problems...")
    problems_zip_path = os.path.join(dataset_path, "problems.json")
    if not os.path.exists(problems_zip_path):
        problems_url = "https://drive.google.com/uc?id=1DG9KBCZU_ZgTg5XfSZsGCJx6SsvWyO-c"
        gdown.download(problems_url, problems_zip_path, quiet=False)
    
    # Extract images
    print("Extracting images...")
    with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(dataset_path, "images"))
    
    # Process problems
    print("Processing problems...")
    with open(problems_zip_path, 'r') as f:
        problems = json.load(f)
    
    # Prepare processed data
    processed_data = {}
    
    for problem_id, problem in tqdm(problems.items()):
        # Only include training examples
        if problem.get('split') != 'train':
            continue
            
        if problem.get('image'):
            image_path = os.path.join(str(problem_id), problem['image'])
            processed_data[f"scienceqa/images/train/{image_path}"] = {
                'question_id': problem_id,
                'caption': format_caption(problem),
                'qa_string': format_qa_string(problem)
            }
    
    # Save processed data
    processed_json_path = os.path.join(dataset_path, "processed_problems.json")
    with open(processed_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} training problems")
    
    # Create processed flag
    with open(processed_flag, 'w') as f:
        f.write("")

def load(cache):
    """Load the processed dataset."""
    dataset_path = os.path.join(cache, "scienceqa")
    processed_json_path = os.path.join(dataset_path, "processed_problems.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    scienceqa_data = {}
    
    for image_path, data in processed_data.items():
        scienceqa_data[image_path] = {
            "image_id": data['question_id'],
            "image_source": "scienceqa",
            "captions": [data['caption']] if data['caption'] else [],
            "bboxes": [],  # No bounding boxes in this dataset
            "QA": data['qa_string']
        }
    
    return scienceqa_data

def info():
    """Return information about the ScienceQA dataset."""
    return {
        "name": "ScienceQA",
        "description": "A dataset of science questions with educational context and images",
        "size": "~500MB",
        "num_datapoints": 6218,
        "num_images": 6218,
        "publication_year": 2023,
        "requires_gpt_conversion": True
    }