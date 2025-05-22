import os
import json
import re
import zipfile
import requests
from utils import plural_to_singular

def parse_how_many_question(question, count):
    """
    Parse a "How many" question and convert it to a statement.
    """
    answer = str(count)
    # Remove punctuation from end of question
    question = re.sub(r'[^\w\s]$', '', question)
    
    # Remove "How many" from the beginning
    question = re.sub(r'^How many\s+', '', question)
    
    # Replace "are there" with empty string
    question = re.sub(r'\bare there\b', '', question)


    # Replace "that are in" with "in"
    question = re.sub(r'\bthat are in\b', 'in', question)

    # Replace "that are" with ""
    question = re.sub(r'\bthat are\b', '', question)
    
    # Replace "are" with "that are" if it's not at the end
    question = re.sub(r'\bare\b(?!$)', 'that are', question)
    
    # Remove "are" if it's at the end
    question = re.sub(r'\bare$', '', question)
    
    # Convert answer to words if it's 0
    if answer == '0':
        answer = 'no'

    # Convert reference to singular if answer is 1
    if answer == '1':
        # Remove "is" from the question when not following "that"
        question = re.sub(r'(?<!that)\bis\b ', '', question)
        # Replace "are" with "is"
        question = re.sub(r'\bare\b', 'is', question)

    # "have" -> "that have"
    question = re.sub(r'\bhave\b', 'that have', question)

    # " do you " -> " that you "
    question = re.sub(r'\bdo you\b', 'that you', question)

    # "does the __ that have" -> "that the __ have" (if plural), "that the __ has" (if singular)
    # "does this __ that have" -> "that the __ have" (if plural), "that the __ has" (if singular)
    if answer == '1':
        question = plural_to_singular(question)
        question = re.sub(r'\bdoes (?:the|this) (.+?) that have\b', 'that the \\1 has', question)
    else:
        question = re.sub(r'\bdoes (?:the|this) (.+?) that have\b', 'that the \\1 have', question)
    
    # "can you see" -> "that you can see"
    question = re.sub(r'\bcan you see\b', 'that you can see', question)
    
    # "can be seen" -> "that can be seen"
    question = re.sub(r'\bcan be seen\b', 'that can be seen', question)

    # Construct the statement
    if answer == 'no':
        statement = f"There are no {question}"
    elif answer == '1':
        statement = f"There is {answer} {question}"
    else:
        statement = f"There are {answer} {question}"

    # Replace "There are no of" with "There are none of"
    statement = re.sub(r'\bThere are no of\b', 'There are none of', statement)
    
    return statement.strip()

def download(cache):
    # Create the dataset directory if it doesn't exist
    dataset_dir = os.path.join(cache, "tallyqa")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Path to indicate the download is complete
    downloaded_flag = os.path.join(dataset_dir, "downloaded")
    
    # Check if the dataset is already downloaded
    if os.path.exists(downloaded_flag):
        print("TallyQA dataset already downloaded.")
        return
    
    # Download the dataset
    url = "https://github.com/manoja328/TallyQA_dataset/blob/46cdc649ec79c3dcc2720ff227ad07d7ee51da6f/tallyqa.zip?raw=true"
    zip_path = os.path.join(cache, "tallyqa.zip")
    
    # Download and save the zip file
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    # Unzip the downloaded file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Remove the zip file after extraction
    os.remove(zip_path)
    
    # Create a file to indicate that the download is complete
    with open(downloaded_flag, "w") as f:
        f.write("Download complete.")
    
    print("TallyQA dataset downloaded and extracted.")

def load(cache):
    dataset_dir = os.path.join(cache, "tallyqa")
    train_file = os.path.join(dataset_dir, "train.json")
    
    # Load the data from the train.json file
    with open(train_file, "r") as f:
        data = json.load(f)
    
    # Prepare the final data structure
    final_data = {}
    
    for item in data:
        image_path = item["image"].replace("train2014/COCO_train2014_", "coco/train2017/")

        # skip val2014 images
        if "val2014" in image_path:
            continue

        if image_path not in final_data:
            final_data[image_path] = {
                "image_id": str(item["image_id"]),
                "image_source": item["data_source"],
                "bboxes": [],    # No bounding boxes provided
                "captions": [],
                "QA": []         # No QA pairs provided
            }
        
        # Add the question-answer pair to the data
        final_data[image_path]["captions"].append(parse_how_many_question(item["question"], item["answer"]))
    
    return final_data

def info():
    return {
        "name": "TallyQA",
        "description": "A dataset designed for answering counting-based questions using images.",
        "size": "54 MB",
        "num_datapoints": 183986,
        "num_images": 100348,
        "publication_year": 2019,
        "requires_gpt_conversion": False
    }
