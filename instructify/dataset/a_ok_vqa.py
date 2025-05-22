import os
import json
import requests
import tarfile

def download(cache):
    dataset_name = "a_ok_vqa"
    dataset_folder = os.path.join(cache, dataset_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    downloaded_flag = os.path.join(dataset_folder, "downloaded")
    if os.path.exists(downloaded_flag):
        print(f"{dataset_name} is already downloaded.")
        return

    # URL and paths
    url = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
    tar_path = os.path.join(dataset_folder, "aokvqa_v1p0.tar.gz")
    
    # Download the dataset
    print(f"Downloading {dataset_name}...")
    response = requests.get(url, stream=True)
    with open(tar_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the tar.gz file
    print(f"Extracting {dataset_name}...")
    with tarfile.open(tar_path, "r:gz") as tar_ref:
        tar_ref.extractall(dataset_folder)
    
    # Clean up the tar file
    os.remove(tar_path)
    
    # Create a 'downloaded' flag
    open(downloaded_flag, 'w').close()
    print(f"{dataset_name} downloaded and extracted successfully.")

def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def load(cache):
    version = 'v1p0'
    split = "train"
    assert split in ['train', 'val', 'test', 'test_w_ans'], "Invalid split name"
    dataset_name = "a_ok_vqa"
    dataset_folder = os.path.join(cache, dataset_name)
    
    # Use the helper function load_aokvqa to load the dataset
    aokvqa_dir = dataset_folder
    dataset = load_aokvqa(aokvqa_dir, split, version)
    
    formatted_data = {}
    for item in dataset:
        image_id = str(item['image_id'])
        image_path = f"coco/train2017/{image_id.zfill(12)}.jpg"
        image_source = "a_ok_vqa"
        bboxes = []  # Assuming no bboxes information is provided in this dataset
        captions = []  # Assuming no captions information is provided in this dataset
        
        # Format the QA field as specified
        question = item['question']
        choices = item['choices']
        correct_choice = choices[item['correct_choice_idx']]
        incorrect_choices = [choice for idx, choice in enumerate(choices) if idx != item['correct_choice_idx']]
        rationale = " ".join(item['rationales']).replace('\n', ' ')  # Concatenate rationales
        
        formatted_data[image_path] = {
            "image_id": image_id,
            "image_source": image_source,
            "bboxes": bboxes,
            "captions": captions,
            "QA": [f"{question} {correct_choice} (not {' nor '.join(incorrect_choices)})\n{rationale}"]
        }
    return formatted_data

def info():
    return {
        "name": "a_ok_vqa",
        "description": "The A-OKVQA dataset, an extension of VQA for knowledge-based question answering.",
        "size": "19 MB",
        "num_datapoints": 16540,
        "num_images": 16540,
        "publication_year": 2022,
        "requires_gpt_conversion": True
    }
