import pandas as pd
import numpy as np
import requests
import zlib
import os
import shelve
import magic
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)

def _file_name(row):
    return "%s/%s_%s" % (row['folder'], row.name, (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

def check_if_downloaded(row):
    fname = _file_name(row)
    return os.path.isfile(fname)

def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    print("Checking existing downloads and generating parts...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:
        
        # First, identify which chunks need processing
        chunks_to_process = []
        for index, i in enumerate(range(0, len(df), chunk_size)):
            chunk = df[i:i + chunk_size]
            
            # Skip if chunk is already in results
            if str(index) in results.keys():
                continue
                
            # Check if any files in this chunk need downloading
            needs_download = False
            for _, row in chunk.iterrows():
                if not check_if_downloaded(row):
                    needs_download = True
                    break
            
            if needs_download:
                chunks_to_process.append((index, chunk))
            else:
                # If all files exist, create result entry with status 200
                processed_chunk = chunk.copy()
                processed_chunk['status'] = 200
                processed_chunk['file'] = processed_chunk.apply(lambda x: _file_name(x), axis=1)
                processed_chunk['mimetype'] = processed_chunk['file'].apply(lambda x: magic.from_file(x, mime=True))
                processed_chunk['size'] = processed_chunk['file'].apply(lambda x: os.stat(x).st_size)
                results[str(index)] = (index, processed_chunk)

        total_chunks = len(chunks_to_process)
        if total_chunks == 0:
            print("All images already downloaded!")
            return
            
        print(f"Found {total_chunks} chunks containing new images to download")
        print(f"Using {processes} processes, {chunk_size} images per chunk")

        pbar = tqdm(total=sum(len(chunk) for _, chunk in chunks_to_process), position=0)
        pbar.desc = "Downloading"
        
        pool_data = ((index, chunk, func) for index, chunk in chunks_to_process)
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return
    
def download_image(row):
    fname = _file_name(row)
    headers = {
        'User-Agent': 'Googlebot-Image/1.0',
        'X-Forwarded-For': '64.18.15.200'
    }
    
    # Skip Already downloaded
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
    except Exception as e:
        row['status'] = 408
        return row
   
    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                response.raw.decode_content = True
                out_file.write(response.content)
            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def download(cache):
    """Download and process the Conceptual Captions dataset."""
    dataset_path = os.path.join(cache, "conceptual_captions")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    processed_flag = os.path.join(dataset_path, "processed")
    if os.path.exists(processed_flag):
        print("Dataset already processed.")
        return
    
    print("Loading Conceptual Captions dataset...")
    dataset = load_dataset("google-research-datasets/conceptual_captions", "labeled", split="train")
    
    # Convert dataset to dataframe
    df = pd.DataFrame({
        'url': dataset['image_url'],
        'caption': dataset['caption'],
        'labels': dataset['labels']
    })
    
    # Set up folder for downloads
    df['folder'] = os.path.join(dataset_path, "images")
    if not os.path.exists(df['folder'].iloc[0]):
        os.makedirs(df['folder'].iloc[0])
    
    # Download images using multiprocessing
    num_processes = 32
    images_per_part = 100
    df_multiprocess(df=df, processes=num_processes, chunk_size=images_per_part, 
                   func=download_image, dataset_name="train")
    
    # Gather results
    print("Collecting results...")
    with shelve.open('train_download_image_%s_results.tmp' % images_per_part) as results:
        keylist = sorted([int(k) for k in results.keys()])
        final_df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    
    # Save metadata only for successful downloads
    successful_downloads = final_df[final_df['status'] == 200]
    metadata = {}
    for _, row in successful_downloads.iterrows():
        filename = os.path.basename(row['file'])
        metadata[filename] = {
            'caption': row['caption'],
            'labels': row['labels']
        }
    
    # Save metadata as JSON
    import json
    with open(os.path.join(dataset_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    # Create processed flag
    with open(processed_flag, 'w') as f:
        f.write("")
    
    print(f"Download complete. Successfully downloaded: {len(successful_downloads)}")
    print(f"Failed downloads: {len(final_df) - len(successful_downloads)}")

def load(cache):
    """Load the processed dataset."""
    dataset_path = os.path.join(cache, "conceptual_captions")
    
    # Load metadata
    with open(os.path.join(dataset_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    cc_data = {}
    
    for filename, meta in metadata.items():
        image_path = f"conceptual_captions/images/{filename}"
        labels_caption = "In this image you can find " + ", ".join(meta['labels'])
        cc_data[image_path] = {
            "image_id": filename,
            "image_source": "conceptual_captions",
            "captions": [meta['caption'], labels_caption],
            "bboxes": [],  # Conceptual Captions doesn't include bounding boxes
            "QA": []      # No QA pairs in this dataset
        }
    
    return cc_data

def info():
    return {
        "name": "Conceptual Captions",
        "description": "A large-scale dataset of image-caption pairs with rich semantic labels",
        "size": "300 GB",
        "num_datapoints": 2000000,
        "num_images": 2000000,
        "publication_year": 2018,
        "requires_gpt_conversion": False
    }