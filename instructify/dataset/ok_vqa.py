import os
import json
import requests
import zipfile

def download(cache):
    dataset_name = "ok_vqa"
    dataset_folder = os.path.join(cache, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Check if the dataset is already downloaded
    downloaded_flag = os.path.join(dataset_folder, "downloaded")
    if os.path.exists(downloaded_flag):
        print(f"{dataset_name} is already downloaded.")
        return
    
    # URLs and paths for the question and annotation files
    question_url = "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip"
    annotation_url = "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip"
    question_zip_path = os.path.join(dataset_folder, "questions.zip")
    annotation_zip_path = os.path.join(dataset_folder, "annotations.zip")
    
    # Download the question file
    print(f"Downloading questions for {dataset_name}...")
    response = requests.get(question_url)
    with open(question_zip_path, 'wb') as f:
        f.write(response.content)
    
    # Download the annotation file
    print(f"Downloading annotations for {dataset_name}...")
    response = requests.get(annotation_url)
    with open(annotation_zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip files
    print(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(question_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_folder)
    with zipfile.ZipFile(annotation_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_folder)
    
    # Clean up the zip files
    os.remove(question_zip_path)
    os.remove(annotation_zip_path)
    
    # Create a 'downloaded' flag
    open(downloaded_flag, 'w').close()
    print(f"{dataset_name} downloaded and extracted successfully.")

def load(cache):
    split = "train"
    assert split in ['train', 'val', 'test'], "Invalid split name"
    dataset_name = "ok_vqa"
    dataset_folder = os.path.join(cache, dataset_name)
    
    # Define paths to the question and annotation JSON files
    question_file = os.path.join(dataset_folder, f"OpenEnded_mscoco_{split}2014_questions.json")
    annotation_file = os.path.join(dataset_folder, f"mscoco_{split}2014_annotations.json")
    
    # Load questions and annotations
    with open(question_file, 'r') as f:
        questions_data = json.load(f)
    with open(annotation_file, 'r') as f:
        annotations_data = json.load(f)
    
    # Create a mapping of question_id to the most common answer
    question_to_answer = {}
    for annotation in annotations_data['annotations']:
        question_id = annotation['question_id']
        answer_counts = {}
        for ans in annotation['answers']:
            answer = ans['answer']
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
        
        # Find the most common answer
        most_common_answer = max(answer_counts, key=answer_counts.get)
        question_to_answer[question_id] = most_common_answer
    
    formatted_data = {}
    for question_item in questions_data['questions']:
        image_id = str(question_item['image_id'])
        image_path = f"coco/train2017/{image_id.zfill(12)}.jpg"
        question_id = question_item['question_id']
        question_text = question_item['question']
        most_common_answer = question_to_answer.get(question_id, "No answer")
        
        QA_entry = f"{question_text} {most_common_answer}"
        
        formatted_data[image_path] = {
            "image_id": image_id,
            "image_source": "ok_vqa",
            "bboxes": [],  # Assuming no bounding boxes information in this dataset
            "captions": [],  # Assuming no captions information in this dataset
            "QA": [QA_entry]
        }
    
    return formatted_data

def info():
    return {
        "name": "ok_vqa",
        "description": "The OK-VQA dataset, a knowledge-based visual question answering dataset.",
        "size": "16 MB",
        "num_questions": 8998,
        "num_images": 8998,
        "publication_year": 2019,
        "requires_gpt_conversion": True
    }
