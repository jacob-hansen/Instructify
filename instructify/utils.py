import os
import re
import json
import requests
from zipfile import ZipFile
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import inflect
import numpy as np
import re

lemmatizer = WordNetLemmatizer()
p = inflect.engine()

def old_format_bboxes(image_data):
    """
    Converts nested image data dictionary into a formatted string of bounding boxes,
    with y-coordinates inverted (0 at bottom, 1 at top).
    
    Args:
        image_data (dict): Dictionary containing image annotations from different sources
        
    Returns:
        str: Formatted string of bounding boxes
    """
    formatted_lines = []
    
    for source, data in image_data.items():
        if data['bboxes']:
            formatted_lines.append(f"\n=== {source} OCR ===")
            
            for bbox in data['bboxes']:
                label = bbox[0]
                # Convert coordinates and invert y values
                x1, y1, x2, y2 = bbox[1:]
                y1, y2 = 1 - y2, 1 - y1  # Invert y coordinates
                coords = [f"{coord:.2f}" for coord in [x1, y1, x2, y2]]
                formatted_lines.append(f"[{label}, {', '.join(coords)}]")
    if len(formatted_lines) == 0:
        return ""
        
    info = "Objects and text are formatted as [label, x1, y1, x2, y2] coordinates, where x ranges from 0 (left) to 1 (right) and y from 0 (bottom) to 1 (top). Where OCR results are given, it may contain errors and inconsistencies."
    return info + "\n".join(formatted_lines) + "\n=== End of OCR ==="
    
def custom_round(value, precision=0.05):
    """
    Rounds a number to the nearest specified decimal precision.
    
    Args:
        value (float): The number to round.
        precision (float): The rounding precision threshold. For example, 0.05 means
                           0.22 and 0.23 would round to 0.20, but 0.26 would round to 0.25.
    
    Returns:
        float: The rounded value based on the specified precision.
    """
    return round(value / precision) * precision
    
def plural_to_singular(phrase):
    """
    Convert plural words to singular in a phrase.
    """
    words = phrase.split(" ")
    for i, word in enumerate(words):
        # Special cases handling
        if "people" in word:
            words[i] = word.replace("people", "person")
        elif "men" in word:
            words[i] = word.replace("men", "man")
        else:
            # General lemmatization
            words[i] = lemmatizer.lemmatize(word, pos='n')
    return " ".join(words)

# Download necessary resources
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def singular_to_plural(phrase):
    """
    Convert singular words to plural in a phrase, ignoring adjectives.
    """
    words = word_tokenize(phrase)
    pos_tags = pos_tag(words)  # POS tagging to identify adjectives
    pluralized_words = []
    
    for i, (word, pos) in enumerate(pos_tags):
        word_is_capitalized = word[0].isupper()
        word_lower = word.lower()
        
        # Change 'a' or 'an' to 'some'
        if word_lower in ['a', 'an']:
            pluralized_words.append('some')
            continue
        
        # Check if word is a noun and not already plural, and avoid adjectives (JJ)
        if pos.startswith('NN') and pos != 'NNS' and lemmatizer.lemmatize(word_lower, pos='n') == word_lower:
            word_plural = p.plural(word_lower)
        else:
            word_plural = word_lower
        
        # Preserve capitalization
        if word_is_capitalized:
            word_plural = word_plural.capitalize()
        
        pluralized_words.append(word_plural)
    
    # Join words, ensuring correct spacing around punctuation
    result = ""
    for i, word in enumerate(pluralized_words):
        if i > 0 and word in [',', '.', '!', '?']:
            result = result.rstrip() + word + ' '
        else:
            result += word + ' '
    
    return result.strip()

def image_id_mapping(cache):
    """
    Get the mapping from image_id to image path for Visual Genome dataset,
    including mappings for COCO IDs
    """
    vg_100k_path = os.path.join(cache, "images", "vg/VG_100K")
    vg_100k_2_path = os.path.join(cache, "images", "vg/VG_100K_2")
    coco_train2017_path = os.path.join(cache, "images", "coco/train2017")
    assert os.path.exists(vg_100k_path), "Images for Visual Genome VG_100K not found. Please download them."
    assert os.path.exists(vg_100k_2_path), "Images for Visual Genome VG_100K_2 not found. Please download them."
    assert os.path.exists(coco_train2017_path), "Images for COCO train2017 not found. Please download them."
    
    # Load image data JSON file
    image_data_path = os.path.join(cache, "visual_genome", "image_data.json")
    assert os.path.exists(image_data_path), "image_data.json not found. Please download it."
    with open(image_data_path, 'r') as f:
        image_data = json.load(f)

    image_mapping = {}

    # Process VG_100K images
    vg_100k_images = os.listdir(vg_100k_path)
    for img in vg_100k_images:
        image_id = int(img.split(".")[0])
        image_mapping[image_id] = f"vg/VG_100K/{img}"

    # Process VG_100K_2 images
    vg_100k_2_images = os.listdir(vg_100k_2_path)
    for img in vg_100k_2_images:
        image_id = int(img.split(".")[0])
        image_mapping[image_id] = f"vg/VG_100K_2/{img}"

    # Process COCO train2017 images and map COCO IDs
    coco_train2017_images = os.listdir(coco_train2017_path)
    for img in coco_train2017_images:
        coco_id = int(img.split(".")[0])
        image_mapping[coco_id] = f"coco/train2017/{img}"

    # Map COCO IDs to Visual Genome IDs
    for img_data in image_data:
        if img_data['coco_id']:
            vg_id = img_data['image_id']
            coco_id = img_data['coco_id']
            if coco_id in image_mapping:
                image_mapping[vg_id] = image_mapping[coco_id]

    return image_mapping

def download_coco_train2017_images(cache):
    """
    Download the COCO train2017 images.
    """
    # Define the target path for downloading images
    images_path = os.path.join(cache, "images/coco/train2017")
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    
    # URL for the COCO train 2017 images
    url = "http://images.cocodataset.org/zips/train2017.zip"
    zip_path = os.path.join(images_path, "train2017.zip")

    # Check if the images are already downloaded
    downloaded_flag = os.path.join(images_path, "downloaded")
    if os.path.exists(downloaded_flag):
        print("COCO train2017 images already downloaded.")
        return

    # Download the images
    print("Downloading COCO train2017 images...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f:
        for data in response.iter_content(1024):
            f.write(data)

    # Unzip the downloaded file
    print("Extracting COCO train2017 images...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(images_path)

    # Remove the zip file
    os.remove(zip_path)

    # Create a flag to indicate successful download
    with open(downloaded_flag, 'w') as f:
        f.write("")

    print("COCO train2017 images downloaded and extracted successfully.")

def box_iou(box1, box2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def mask_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) of two masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union
    
def extract_parentheses(label):
    match = re.search(r'\((.*?)\)', label)
    return match.group(1) if match else None

# Helper function to remove content inside parentheses
def remove_parentheses(label):
    return re.sub(r'\s*\(.*?\)', '', label).strip()

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # Initialize previous row of distances
    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # Cost of insertion
            deletions = current_row[j] + 1        # Cost of deletion
            substitutions = previous_row[j] + (c1 != c2)  # Cost of substitution
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def is_relationship(label):
    """Check if the label is a relationship statement."""
    return any(phrase in label for phrase in RELATIONSHIP_PHRASES)

RELATIONSHIP_PHRASES = [
        "next to", "on top of", "below", "above", "beside", "near",
        "under", "over", "around", "in front of", "behind"
    ]

def merge_labels(label_list, distance_percentage=0.2):
    """
    Merge similar labels based on relationship status, substring containment,
    and Levenshtein distance as a distance_percentage of label lengths.
    """
    merged_labels = []
    remaining_labels = label_list.copy()

    while remaining_labels:
        base_label = remaining_labels.pop(0)
        base_is_relationship = is_relationship(base_label)
        labels_to_merge = []

        for label in remaining_labels:
            label_is_relationship = is_relationship(label)

            # Skip if one is a relationship statement and the other is not
            if base_is_relationship != label_is_relationship:
                continue

            # Check for substring containment
            if base_label in label or label in base_label:
                # Keep the longer label
                base_label = label if len(label) > len(base_label) else base_label
                labels_to_merge.append(label)
            else:
                # Calculate Levenshtein distance
                distance = levenshtein_distance(base_label, label)

                # Calculate allowed max distance as a distance_percentage of the average length
                avg_length = (len(base_label) + len(label)) / 2
                allowed_distance = max(1, int(distance_percentage * avg_length))

                if distance <= allowed_distance:
                    # Keep the shorter label
                    base_label = label if len(label) < len(base_label) else base_label
                    labels_to_merge.append(label)

        # Remove merged labels from the remaining labels
        remaining_labels = [label for label in remaining_labels if label not in labels_to_merge]
        merged_labels.append(base_label)

    return merged_labels

def format_labels(merged_labels):
    """
    Format merged labels into a single string with the primary label and additional info.
    """
    if len(merged_labels) == 1:
        return merged_labels[0]
    
    primary_label = merged_labels[0]
    additional_labels = ", ".join([i.strip() for i in merged_labels[1:]])
    return f"{primary_label} ({additional_labels})"

def merge_bboxes(bboxes, iou_threshold=0.9, format_the_labels=True):
    merged = []
    for bbox in bboxes:
        labels, x1, y1, w, h = bbox
        if type(labels) == str:
            labels = [labels]
        x2, y2 = x1 + w, y1 + h
        current_box = [labels, x1, y1, x2, y2]  # Store labels as a list
        
        merged_with_existing = False
        for i, merged_box in enumerate(merged):
            if box_iou(current_box[1:], merged_box[1:]) > iou_threshold:
                # Merge labels by extending the label list
                merged[i][0].extend(current_box[0])
                merged[i][1] = min(merged[i][1], x1)
                merged[i][2] = min(merged[i][2], y1)
                merged[i][3] = max(merged[i][3], x2)
                merged[i][4] = max(merged[i][4], y2)
                merged_with_existing = True
                break
        
        if not merged_with_existing:
            merged.append(current_box)
    
    # Convert back to x1, y1, w, h format and process labels
    for i, box in enumerate(merged):
        box[3] = box[3] - box[1]  # width
        box[4] = box[4] - box[2]  # height
        # Merge similar labels and keep the appropriate one
        merged_labels = merge_labels(box[0])
        # Format the final label string
        if format_the_labels:
            merged[i][0] = format_labels(merged_labels)
        else:
            merged[i][0] = merged_labels
    return merged

def labels_are_plural_of_each_other(labels1, labels2):
    for l1 in labels1:
        for l2 in labels2:
            # Check if l1's plural form equals l2 and they are not the same word
            if singular_to_plural(l1) == l2 and l1 != l2:
                return True
            # Check if l2's plural form equals l1 and they are not the same word
            if singular_to_plural(l2) == l1 and l1 != l2:
                return True
    return False

def masked_merge(input_boxes, masks, iou_threshold=0.6, format_the_labels=True):
    """
    Merge bounding boxes and masks based on both bounding box IoU and mask IoU,
    while ensuring boxes with labels that are plural forms of each other are not merged.

    Args:
        input_boxes (list): List of bounding boxes, each in [label, x1, y1, x2, y2] format.
        masks (list): List of binary masks corresponding to the input_boxes.
        iou_threshold (float): Threshold for both box IoU and mask IoU to consider merging.

    Returns:
        merged_boxes (list): List of merged bounding boxes in [label, x1, y1, x2, y2] format.
        merged_masks (list): List of merged masks corresponding to the merged_boxes.
    """
    merged = []

    for bbox, mask in zip(input_boxes, masks):
        labels, x1, y1, x2, y2 = bbox
        if isinstance(labels, str):
            labels = [labels]
        current_box = {'labels': labels, 'coords': [x1, y1, x2, y2], 'mask': mask}
        merged_with_existing = False

        for merged_box in merged:
            # Check if any labels are plural forms of each other
            if labels_are_plural_of_each_other(current_box['labels'], merged_box['labels']):
                continue  # Skip merging with this box

            # Compute box IoU between current bounding box and existing merged bounding box
            box_iou_value = box_iou(current_box['coords'], merged_box['coords'])
            # Compute mask IoU between current mask and existing merged mask
            mask_iou_value = mask_iou(current_box['mask'], merged_box['mask'])

            # Merge if both IoU values exceed the threshold
            if box_iou_value > iou_threshold and mask_iou_value > iou_threshold:
                # Merge labels
                merged_box['labels'].extend(current_box['labels'])
                # Update bounding box coordinates
                merged_box['coords'][0] = min(merged_box['coords'][0], current_box['coords'][0])
                merged_box['coords'][1] = min(merged_box['coords'][1], current_box['coords'][1])
                merged_box['coords'][2] = max(merged_box['coords'][2], current_box['coords'][2])
                merged_box['coords'][3] = max(merged_box['coords'][3], current_box['coords'][3])
                # Update mask by combining with logical OR
                merged_box['mask'] = np.logical_or(merged_box['mask'], current_box['mask'])
                merged_with_existing = True
                break

        if not merged_with_existing:
            merged.append(current_box)

    # Prepare the final merged_boxes and merged_masks lists
    merged_boxes = []
    merged_masks = []
    for merged_box in merged:
        labels = merged_box['labels']
        x1, y1, x2, y2 = merged_box['coords']
        # Merge and format labels
        merged_labels = merge_labels(labels)

        if format_the_labels:
            final_label = format_labels(merged_labels)
        else:
            final_label = merged_labels
        merged_boxes.append([final_label, x1, y1, x2, y2])
        merged_masks.append(merged_box['mask'])
    return merged_boxes, merged_masks