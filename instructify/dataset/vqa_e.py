import os
import json
import gdown
from collections import Counter
from tqdm import tqdm

def process_qa_pair(qa_item):
    """Process a single QA pair, returning None if invalid."""
    question = qa_item['question']
    answers = qa_item['answers']
    explanation = qa_item['explanation'][0] if isinstance(qa_item['explanation'], list) else qa_item['explanation']
    
    # Handle different answer types
    if qa_item['answer_type'] == 'yes/no':
        # Count yes/no responses
        answer_counts = Counter([a.lower() for a in answers])
        if answer_counts.get('yes', 0) > 0 and answer_counts.get('no', 0) > 0:
            # Discard contradictory yes/no answers
            return None
        
        # Use majority vote
        final_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        
    elif qa_item['answer_type'] == 'number':
        # Try to convert all answers to numbers and take the mode
        try:
            numeric_answers = [float(a) for a in answers if str(a).replace('.','').isdigit()]
            if not numeric_answers:
                return None
            final_answer = str(Counter(numeric_answers).most_common(1)[0][0])
        except ValueError:
            return None
            
    else:  # other/default case
        # Use the most common answer
        answer_counts = Counter(answers)
        final_answer = answer_counts.most_common(1)[0][0]
    
    return {
        'question': question,
        'answer': final_answer,
        'explanation': explanation
    }

def download(cache):
    """Download and process the VQA dataset."""
    dataset_path = os.path.join(cache, "vqa_e")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    # Download dataset
    print("Downloading VQA dataset...")
    json_path = os.path.join(dataset_path, "vqa_e.json")
    if not os.path.exists(json_path):
        url = "https://drive.google.com/uc?id=1CXogPObRixI1iR51T2px-Q75jdnhByCX"
        gdown.download(url, json_path, quiet=False)
    
    # Process QA pairs
    print("Processing QA pairs...")
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
    
    # Group QAs by image_id
    processed_data = {}
    for qa_item in tqdm(qa_data):
        img_id = qa_item['img_id']
        processed_qa = process_qa_pair(qa_item)
        
        if processed_qa is not None:
            padded_id = str(img_id).zfill(12)
            image_path = f"coco/train2017/{padded_id}.jpg"
            
            if image_path not in processed_data:
                processed_data[image_path] = {
                    'image_id': padded_id,
                    'QAs': []
                }
            
            processed_data[image_path]['QAs'].append(processed_qa)
    
    # Save processed data
    processed_json_path = os.path.join(dataset_path, "processed_qa.json")
    with open(processed_json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} images with QA pairs")
    
    # Create processed flag
    with open(processed_flag, 'w') as f:
        f.write("")

def load(cache):
    """Load the processed dataset."""
    dataset_path = os.path.join(cache, "vqa_e")
    processed_json_path = os.path.join(dataset_path, "processed_qa.json")
    
    with open(processed_json_path, 'r') as f:
        processed_data = json.load(f)
    
    vqa_data = {}
    
    for image_path, data in processed_data.items():
        vqa_data[image_path] = {
            "image_id": data['image_id'],
            "image_source": "coco",
            "captions": [],  # No captions in this dataset
            "bboxes": [],   # No bounding boxes in this dataset
            "QA": [ qa['question'] + " " + qa['answer'] + " (" + qa['explanation'] + ")" for qa in data['QAs'] ]
        }
    
    return vqa_data

def info():
    """Return information about the VQA dataset."""
    return {
        "name": "VQA Explained",
        "description": "A dataset of visual question-answer pairs with explanations",
        "size": "61MB",
        "num_datapoints": "156705",
        "num_images": "69375",
        "publication_year": 2023,
        "requires_gpt_conversion": True
    }