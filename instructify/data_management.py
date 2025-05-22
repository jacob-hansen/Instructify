import os
import random
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from typing import List, Dict
import shutil
from multiprocessing import Pool

# Load environment variables
load_dotenv()

class DatasetError(Exception):
    """Custom exception for dataset-related errors."""
    pass

class DatasetManager:
    LOADED_DATA = None

    def __init__(self, run_id="0", max_workers: int = 1, already_processed: List[str] = None):
        self.cache_dir = os.getenv("INSTRUCTIFY_CACHE")
        self.run_id = run_id
        if not self.cache_dir:
            raise DatasetError("INSTRUCTIFY_CACHE environment variable is not set.")
        self.max_workers = max_workers
        self.already_processed = set(already_processed or [])
        
        # Initialize available images set
        self._available_images = None
        self._initialize_available_images()
    
    def _initialize_available_images(self):
        """Initialize the set of available images by checking against existing files."""
        if self.LOADED_DATA is None:
            return

        # Convert LOADED_DATA keys to a set for O(1) lookup
        all_images = set(self.LOADED_DATA.keys())
        
        # Remove already processed images
        available = all_images - self.already_processed
        
        # Check existing files in one batch
        save_dir = os.path.join(self.cache_dir, "save")
        if os.path.exists(save_dir):
            existing_files = {f.split('-')[0] for f in os.listdir(save_dir) 
                            if f.endswith(f'-{self.run_id}.jsonl')}
            available -= existing_files
        
        # Convert to list and shuffle once
        self._available_images = list(available)
        random.shuffle(self._available_images)

    def download(self, dataset_names: List[str]):
        """
        Download the specified datasets to the cache directory.
        
        E.g. manager.download(["coco_captions", "lvis"])
        """
        # print which datasets are already downloaded
        not_downloaded = []
        for dataset_name in dataset_names:
            dataset_folder = os.path.join(self.cache_dir, dataset_name)
            downloaded_flag = os.path.join(dataset_folder, "downloaded")
            if not os.path.exists(downloaded_flag):
                not_downloaded.append(dataset_name)
            else:
                print(f"{dataset_name} is already downloaded.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_dataset, name) for name in not_downloaded]
            for future in futures:
                future.result()  # This will raise any exceptions that occurred during download

    def load(self, dataset_names: List[str]) -> Dict[str, Dict[str, any]]:
        """
        Load the specified datasets into memory.

        E.g. manager.load(["coco_captions", "lvis"])
        """
        # first check if all datasets are available for loading
        for dataset_name in dataset_names:
            dataset_folder = os.path.join(self.cache_dir, dataset_name)
            if not os.path.exists(dataset_folder):
                raise DatasetError(f"Dataset '{dataset_name}' not found. Please download it first.")
                
        merged_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._load_dataset, name) for name in dataset_names]
            for future in futures:
                dataset_name, dataset = future.result()
                self._merge_dataset(merged_data, dataset_name, dataset)
        if self.LOADED_DATA is None:
            self.LOADED_DATA = merged_data
        else:
            for dataset_name, dataset in merged_data.items():
                self._merge_dataset(self.LOADED_DATA, dataset_name, dataset)
        return merged_data
    
    def set_data(self, data: Dict[str, Dict[str, any]]):
        """
        Set the loaded data directly.

        E.g. manager.set_data({"coco_captions": {"coco/img_1.png": {"caption": "A cat"}}})
        """
        self.LOADED_DATA = data
        
    def drop_imageset(self, image_folder: str):
        """
        Drop all data related to a specific image folder.

        E.g. manager.drop_imageset("ade20k")
        """
        initial_len = len(self.LOADED_DATA)
        if self.LOADED_DATA is None:
            raise DatasetError("No dataset loaded.")
        self.LOADED_DATA = {k: v for k, v in self.LOADED_DATA.items() if image_folder not in k}
        if len(self.LOADED_DATA) == initial_len:
            print(f"WARNING: No images dropped related to {image_folder}")
        return self.LOADED_DATA
    
    def drop_dataset(self, dataset_name: str, only_when_alone: bool = False):
        """
        Drop all data related to a specific dataset.

        E.g. manager.drop_dataset("coco_captions")
        """
        if self.LOADED_DATA is None:
            raise DatasetError("No dataset loaded.")
        data_was_dropped = False
        new_data = {}
        for image_path, data in self.LOADED_DATA.items():
            image_new_data = {k: v for k, v in data.items() if k != dataset_name}
            if len(image_new_data) != len(data):
                data_was_dropped = True
            if len(image_new_data) > 0:
                if only_when_alone:
                    new_data[image_path] = data
                else:
                    new_data[image_path] = image_new_data
        if not data_was_dropped:
            print(f"WARNING: No data dropped related to {dataset_name}")
        self.LOADED_DATA = new_data
        return self.LOADED_DATA

    def sample(self, n: int):
        """
        Sample n random images from the loaded dataset.

        E.g. manager.sample(10)
        """
        if self.LOADED_DATA is None:
            raise DatasetError("No dataset loaded.")
        keys = random.sample(list(self.LOADED_DATA.keys()), n)
        return {k: self.LOADED_DATA[k] for k in keys}
    
    def reserve(self, n: int = 1, run_id: str = None):
        """
        Sample images that haven't been processed yet, more efficiently.
        
        :param n: Number of images to sample
        :param run_id: Optional run_id to override the default
        :return: Dictionary of sampled images and their data
        """
        if self.LOADED_DATA is None:
            raise DatasetError("No dataset loaded.")
            
        if run_id is not None and run_id != self.run_id:
            # If run_id changes, we need to reinitialize available images
            self.run_id = run_id
            self._initialize_available_images()
            
        if self._available_images is None:
            self._initialize_available_images()
            
        # Take the first n available images
        reserved = []
        save_dir = os.path.join(self.cache_dir, "save")
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate how many images we can actually reserve
        n_available = min(n, len(self._available_images))
        if n_available == 0:
            return {}
            
        # Reserve the images
        reserved = self._available_images[:n_available]
        self._available_images = self._available_images[n_available:]
        
        # Batch create the reservation files
        for img in reserved:
            result_file = os.path.join(save_dir, f"{img}-{self.run_id}.jsonl")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, 'w') as f:
                f.write("")
            self.already_processed.add(img)
            
        return {k: self.LOADED_DATA[k] for k in reserved}

    def cache(self, name: str):
        """
        Cache the loaded dataset to the output json file to self.cache_dir + name.

        E.g. manager.cache("base_data.json")
        """
        if self.LOADED_DATA is None:
            raise DatasetError("No dataset loaded.")
        with open(os.path.join(self.cache_dir, name), "w") as f:
            json.dump(self.LOADED_DATA, f)

    def load_cache(self, name: str):
        """
        Load the dataset from the output json file.

        E.g. manager.load_from_cache("base_data.json")
        """
        if not os.path.exists(os.path.join(self.cache_dir, name)):
            raise DatasetError(f"Cache file '{name}' not found.")
        with open(os.path.join(self.cache_dir, name), "r") as f:
            self.LOADED_DATA = json.load(f)
        return self.LOADED_DATA
    
    def cache_image_result(self, image_name: str, result_json: Dict, run_id: str = None):
        """
        Cache the result for a specific image.

        :param image_name: The name of the image (e.g. 'coco/img_1.png')
        :param result_json: The JSON result to cache
        :param run_id: Optional run_id to override the default
        """
        if run_id is None:
            run_id = self.run_id

        # Create the directory structure if it doesn't exist
        dir_path = os.path.join(self.cache_dir, "save", os.path.dirname(image_name))
        os.makedirs(dir_path, exist_ok=True)

        # Create or append to the JSONL file
        jsonl_path = os.path.join(self.cache_dir, "save", f"{image_name}-{run_id}.jsonl")
        with open(jsonl_path, 'a') as f:
            json.dump(result_json, f)
            f.write('\n')

        # Add the result to the loaded data
        if self.LOADED_DATA is not None and image_name in self.LOADED_DATA:
            self.LOADED_DATA[image_name][f"result_{run_id}"] = result_json
        
    def cache_image_result(self, image_name: str, result_json: Dict, run_id: str = None):
        """
        Cache the result for a specific image.

        :param image_name: The name of the image (e.g. 'coco/img_1.png')
        :param result_json: The JSON result to cache
        :param run_id: Optional run_id to override the default
        """
        if run_id is None:
            run_id = self.run_id

        # Create the directory structure if it doesn't exist
        dir_path = os.path.join(self.cache_dir, "save", os.path.dirname(image_name))
        os.makedirs(dir_path, exist_ok=True)

        # Create or append to the JSONL file
        jsonl_path = os.path.join(self.cache_dir, "save", f"{image_name}-{run_id}.jsonl")
        with open(jsonl_path, 'a') as f:
            json.dump(result_json, f)
            f.write('\n')

        # Add the result to the loaded data
        if self.LOADED_DATA is not None and image_name in self.LOADED_DATA:
            self.LOADED_DATA[image_name][f"result_{run_id}"] = result_json

        # Update the indicator file
        indicator_dir = os.path.join(self.cache_dir, "save", "indicators")
        os.makedirs(indicator_dir, exist_ok=True)
        indicator_file_path = os.path.join(indicator_dir, f"indicator_{run_id}.txt")
        with open(indicator_file_path, 'w') as f:
            f.write('')

    def collect_results(self, run_id: str = None):
        """
        Collect all results for a given run_id.

        :param run_id: Optional run_id to override the default. If "ALL", collect all results.
        :return: Dictionary mapping image names to results
        """
        if run_id is None:
            run_id = self.run_id

        cache_file_name = f"{run_id}_results_cache.json"
        cache_file_path = os.path.join(self.cache_dir, "save", cache_file_name)
        indicator_dir = os.path.join(self.cache_dir, "save", "indicators")
        indicator_file_path = os.path.join(indicator_dir, f"indicator_{run_id}.txt")

        # Check if cache file and indicator file exist
        if os.path.exists(cache_file_path) and os.path.exists(indicator_file_path):
            cache_mtime = os.path.getmtime(cache_file_path)
            indicator_mtime = os.path.getmtime(indicator_file_path)

            if cache_mtime >= indicator_mtime:
                # Cache is up to date; load results from cache
                with open(cache_file_path, 'r') as f:
                    all_results = json.load(f)
                return all_results
        
        # Else, need to reload all files and overwrite cache
        all_results = {}
        print("Loading results from files, may take awhile...")

        save_dir = os.path.join(self.cache_dir, "save")
        for root, _, files in os.walk(save_dir):
            for file in files:
                # Skip the cache file and indicator files
                if file == cache_file_name or file.startswith('indicator_'):
                    continue

                # Check if the file corresponds to the run_id or if we're collecting all runs
                if run_id == "ALL" or file.endswith(f"-{run_id}.jsonl"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        # Read all JSON lines from the file
                        results = [json.loads(line) for line in f]

                    # Extract the image name from the file path
                    rel_file_path = os.path.relpath(file_path, save_dir)
                    # Remove the run_id and extension from the file name to get the image name
                    if run_id == "ALL":
                        # Remove the last suffix that matches '-<run_id>.jsonl'
                        base_name = rel_file_path.rsplit('-', 1)[0]
                        image_name = base_name
                    else:
                        suffix = f"-{run_id}.jsonl"
                        if rel_file_path.endswith(suffix):
                            image_name = rel_file_path[:-len(suffix)]
                        else:
                            image_name = rel_file_path[:-len('.jsonl')]

                    # Normalize path separators
                    image_name = image_name.replace(os.sep, '/')

                    # If multiple results per image are possible, store them in a list
                    if image_name in all_results:
                        all_results[image_name].extend(results)
                    else:
                        all_results[image_name] = results

        # Save the combined data into the cache file
        with open(cache_file_path, 'w') as f:
            json.dump(all_results, f)

        return all_results
    
    def count_results(self, run_id: str = None) -> dict:
        """
        Quickly count the number of result files for a given run_id without loading their contents.

        :param run_id: Optional run_id to override the default. If "ALL", count all results.
        :return: 'total_files' the total number of result files
        """
        if run_id is None:
            run_id = self.run_id

        save_dir = os.path.join(self.cache_dir, "save")
        file_count = 0

        for root, _, files in os.walk(save_dir):
            for file in files:
                # Skip cache and indicator files
                if file.endswith('_results_cache.json') or file.startswith('indicator_'):
                    continue

                # Check if the file corresponds to the run_id or if we're collecting all runs
                if run_id == "ALL" or file.endswith(f"-{run_id}.jsonl"):
                    file_count += 1

        return file_count

    @staticmethod
    def _remove_file(file_path):
        try:
            os.remove(file_path)
            return file_path
        except FileNotFoundError:
            return None

    def clean(self, run_id: str = None, workers: int = 1, empty_only: bool = True):
        """
        Remove files of a given run_id using multiprocessing.

        :param run_id: The run_id to clean. If None, uses the default run_id.
        :param workers: Number of worker processes to use. If None, uses os.cpu_count().
        :param empty_only: If True, only remove files with content length < 3 characters.
        """
        if run_id is None:
            run_id = self.run_id

        save_dir = os.path.join(self.cache_dir, "save")
        if not os.path.exists(save_dir):
            print(f"No data found in save directory: {save_dir}")
            return

        def find_files(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(f"-{run_id}.jsonl"):
                        yield os.path.join(root, file)

        # Find all files to potentially remove
        files_to_check = list(find_files(save_dir))

        # Use multiprocessing to check and remove files
        with Pool(workers) as pool:
            removed_files = pool.starmap(self._check_and_remove_file, [(file, empty_only) for file in files_to_check])

        removed_count = sum(1 for file in removed_files if file is not None)
        print(f"Cleaned {removed_count} files for run_id: {run_id}")

        # Remove empty directories
        for root, dirs, files in os.walk(save_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

        # Optionally, remove the entire save directory if it's empty
        if not os.listdir(save_dir):
            shutil.rmtree(save_dir)
            print(f"Removed empty save directory: {save_dir}")

        # Update the run id indicator file
        indicator_dir = os.path.join(save_dir, "indicators")
        indicator_file_path = os.path.join(indicator_dir, f"indicator_{run_id}.txt")
        with open(indicator_file_path, 'w') as f:
            f.write('')

    def _check_and_remove_file(self, file_path: str, empty_only: bool) -> str:
        """
        Check if a file should be removed and remove it if necessary.

        :param file_path: Path to the file to check and potentially remove.
        :param empty_only: If True, only remove files with minimal content length or 
                        files that contain only an empty list '[]'.
        :return: The path of the removed file, or None if not removed.
        """
        if empty_only:
            try:
                # Get file stats
                file_info = os.stat(file_path)
                file_size = file_info.st_size

                # check if first 5 characters are "Error"
                with open(file_path, 'r') as f:
                    first_char = f.read(1)
                    if first_char != '[':
                        os.remove(file_path)
                        return file_path
                
                # Check if file is very small
                if file_size < 3:
                    os.remove(file_path)
                    return file_path
                else:
                    return None
            except (IOError, OSError):
                return None

        try:
            os.remove(file_path)
            return file_path
        except OSError:
            return None
        
    def _download_dataset(self, dataset_name: str):
        try:
            module = importlib.import_module(f"dataset.{dataset_name}")
            if not hasattr(module, 'download'):
                raise DatasetError(f"Dataset '{dataset_name}' does not have a download function.")
            module.download(self.cache_dir)
            print(f"Downloaded {dataset_name}")
        except ImportError:
            raise DatasetError(f"Dataset '{dataset_name}' not found or failed to import.")
        except Exception as e:
            raise DatasetError(f"Error downloading dataset '{dataset_name}': {str(e)}")

    def _load_dataset(self, dataset_name: str):
        try:
            module = importlib.import_module(f"dataset.{dataset_name}")
            if not hasattr(module, 'load'):
                raise DatasetError(f"Dataset '{dataset_name}' does not have a load function.")
            dataset = module.load(self.cache_dir)
            print(f"Loaded {dataset_name}")
            return dataset_name, dataset
        except ImportError:
            raise DatasetError(f"Dataset '{dataset_name}' not found or failed to import.")
        except Exception as e:
            raise DatasetError(f"Error loading dataset '{dataset_name}': {str(e)}")

    def _merge_dataset(self, merged_data: Dict[str, Dict[str, any]], dataset_name: str, dataset: any):
        for image_path, data in dataset.items():
            if image_path not in merged_data:
                merged_data[image_path] = {}
            merged_data[image_path][dataset_name] = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.LOADED_DATA is None or not self.LOADED_DATA:
            return "DatasetManager(LOADED_DATA=None)"
        dataset_info = {}
        for img, data in self.LOADED_DATA.items():
            for dataset_name in data.keys():
                if dataset_name not in dataset_info:
                    dataset_info[dataset_name] = 0
                dataset_info[dataset_name] += 1
            
        dataset_info_str = "\t" + "\n\t".join([f"{k}: {v} images" for k, v in dataset_info.items()])
        return f"DatasetManager ({len(self.LOADED_DATA)} images) with loaded datasets:\n{dataset_info_str}"
    
    def __iter__(self):
        """
        While reserve images are available, yield them.
        """
        while True:
            reserved = self.reserve()
            if not reserved:
                break
            yield reserved