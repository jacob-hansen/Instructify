import os
import json
import pandas as pd
import requests
from zipfile import ZipFile

def download(cache):
    # Directory where the dataset files will be stored
    dataset_dir = os.path.join(cache, "vqa_v2")
    os.makedirs(dataset_dir, exist_ok=True)

    # URLs for the dataset files
    annotation_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
    question_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"

    # File names for the downloads
    annotation_zip_file = os.path.join(dataset_dir, "v2_Annotations_Train_mscoco.zip")
    question_zip_file = os.path.join(dataset_dir, "v2_Questions_Train_mscoco.zip")
    
    # Download the annotations file
    if not os.path.exists(annotation_zip_file):
        print(f"Downloading {annotation_url} to {annotation_zip_file}...")
        response = requests.get(annotation_url)
        with open(annotation_zip_file, "wb") as f:
            f.write(response.content)

    # Download the questions file
    if not os.path.exists(question_zip_file):
        print(f"Downloading {question_url} to {question_zip_file}...")
        response = requests.get(question_url)
        with open(question_zip_file, "wb") as f:
            f.write(response.content)

    # Unzip the files
    with ZipFile(annotation_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    with ZipFile(question_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    # Create a file to indicate the dataset has been downloaded
    with open(os.path.join(dataset_dir, "downloaded"), "w") as f:
        f.write("")

def load(cache):
    dataset_dir = os.path.join(cache, "vqa_v2")
    annotations_file = os.path.join(dataset_dir, "v2_mscoco_train2014_annotations.json")
    questions_file = os.path.join(dataset_dir, "v2_OpenEnded_mscoco_train2014_questions.json")
    
    # Load JSON data for annotations and questions
    with open(annotations_file) as f:
        vqa_annotations = json.load(f)["annotations"]
    with open(questions_file) as f:
        vqa_questions = json.load(f)["questions"]
    
    # Convert to DataFrames
    annotations_df = pd.DataFrame(vqa_annotations)
    questions_df = pd.DataFrame(vqa_questions)

    # Create a mapping from question_id to question text
    question_mapping = questions_df.set_index('question_id')['question'].to_dict()
    
    # Initialize a dictionary to hold the formatted data
    localized_narratives = {}

    # Group by 'image_id' and process into the required format
    for image_id, group in annotations_df.groupby('image_id'):
        image_path = f"coco/train2017/{str(image_id).zfill(12)}.jpg"
        
        # Initialize the dictionary entry if it doesn't exist
        if image_path not in localized_narratives:
            localized_narratives[image_path] = {
                "image_id": str(image_id),
                "image_source": "coco_vqa",
                "bboxes": [],  # No bounding boxes
                "captions": [],  # No captions
                "QA": []
            }

        # Add question-answer pairs to the QA list
        group.apply(lambda row: localized_narratives[image_path]["QA"].append(question_mapping[row["question_id"]] + " " + row["multiple_choice_answer"]), axis=1)

    return localized_narratives

def info():
    return {
        "name": "VQAv2",
        "description": "VQAv2 dataset for visual question answering using the COCO images.",
        "size": "407 MB",
        "num_datapoints": 443757,
        "num_images": 82783,
        "publication_year": 2017,
        "requires_gpt_conversion": True
    }
