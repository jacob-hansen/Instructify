import os
os.environ["INSTRUCTIFY_CACHE"] = "/dccstor/mm-instruct1/jacobspace/instructify/instructify/data"
assert 'INSTRUCTIFY_CACHE' in os.environ, "INSTRUCTIFY_CACHE environment variable must be set with the path to the cache directory"

import shutil
import argparse
import asyncio
import time
import torch
from PIL import Image

from generation import get_async_model
from prompt_manager import PromptManager
from data_management import DatasetManager
from utils import old_format_bboxes
from examples import PROMPT_DISTRIBUTIONS

async def main_async(args):
    PROMPT_DISTRIBUTION = PROMPT_DISTRIBUTIONS.get(args.prompt_template, None)
    if PROMPT_DISTRIBUTION is None:
        raise ValueError(f"Prompt distribution {args.prompt_template} not found, available prompt distributions are {list(PROMPT_DISTRIBUTIONS.keys())}")

    # Setup remains the same...
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs available for tensor parallelism.")

    # Initialize objects
    model_callable = get_async_model(
        args.model,
        gpu_memory_utilization=args.vllm_gpu_mem_fraction,
        engine_args={
            "tensor_parallel_size": num_gpus,
            "disable_custom_all_reduce": True,
            "max_model_len": args.max_sequence_length
        }
    )
    
    from conversion.box import HierarchicalObjectOrganizer # import locally to avoid conflicts with vllm
    if args.disable_bbox_tree:
        organizer = None
    else:
        organizer = HierarchicalObjectOrganizer()
    prompt_manager = PromptManager()

    # Data Manager (where data is saved)
    os.makedirs(args.output_path, exist_ok=True)
    data_manager = DatasetManager(args.output_path, max_workers=8)
    data = data_manager.load_cache(args.dataset_name)

    # Loop to process images
    async def process_image():
        nonlocal last_success_time
        while True:
            image_dataset = data_manager.reserve(10, run_id=args.run_id)
            if image_dataset is None:
                continue
            
            for img in image_dataset:
                image_data = image_dataset[img]

                # Make sure image exists
                img_path = os.path.join(os.environ['INSTRUCTIFY_CACHE'], img)
                if not os.path.exists(img_path):
                    print(f"Image {img} not found")
                    continue
                
                # Process information and image
                information = []
                qa_sections = []
                for dataset in data[img]:
                    if "captions" in data[img][dataset]:
                        information += data[img][dataset]["captions"]
                    if "QA" in data[img][dataset] and len(data[img][dataset]["QA"]) > 0:
                        qa_sections.extend(data[img][dataset]["QA"])

                if len(qa_sections) > 0:
                    information += await prompt_manager.run_prompt("conversion/qa", data[img], model_callable)

                if not args.disable_bbox_tree:
                    box_str = await organizer.image_data_conversion(
                        img_path,
                        image_data,
                        include_box_label=True,
                        depth_calculation=True
                    )

                    if len(box_str) > 20:
                        box_captioned = await prompt_manager.run_prompt("conversion.to_caption", box_str, model_callable, max_retries=1)
                        if not box_captioned:
                            print(f"Failed to process box caption for image {img}")
                            box_captioned = ""
                    else:
                        box_captioned = ""
                else:
                    old_box_format = old_format_bboxes(image_data)
                    if len(old_box_format) > 0:
                        box_captioned = [old_box_format,]
                    else:
                        box_captioned = []
                
                information += box_captioned

                result = await prompt_manager.process(information, model_callable, PROMPT_DISTRIBUTION, max_count=args.max_sample_count, filtering_enabled=(not args.disable_filtering), min_information_length=10)
                
                if result:
                    data_manager.cache_image_result(img, result, run_id=args.run_id)
                    last_success_time = time.time()  # Update timestamp on successful cache
                    print("Finished processing image", img)
                else:
                    print(f"Failed to process image {img}, result is {result}")

    # Global variable to track last successful cache
    last_success_time = time.time()
    async def monitor_progress():
        while True:
            await asyncio.sleep(60)  # Check every minute
            if time.time() - last_success_time > 600:  # 10 minutes
                print("No progress detected for 10 minutes. Exiting.")
                os._exit(1)  # Force quit the program

    # Run workers and monitor
    workers = [asyncio.create_task(process_image()) for _ in range(args.num_workers)]
    monitor = asyncio.create_task(monitor_progress())
    await asyncio.gather(*workers, monitor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with captioning and bounding box analysis using language models")
    parser.add_argument("--run_id", type=str, help="Unique identifier to track this processing run in the cache")
    parser.add_argument("--dataset_name", type=str, help="Name of JSON dataset file previously cached via DatasetManager.cache()")
    parser.add_argument("--num_workers", type=int, default=80, help="Number of concurrent processing workers, usually ~40 per GPU is sufficient")
    parser.add_argument("--model", type=str, default="google/gemma-2-27b-it", help="HuggingFace model ID for language processing")
    parser.add_argument("--vllm_gpu_mem_fraction", type=float, default=0.85, help="Fraction of GPU memory to allocate for language model (0.0-0.85), need space for SAM2 and Depth Anything V2 if not disabled")
    parser.add_argument("--output_path", type=str, default="./data", help="Directory to save processing files, use dataset_manager to load from this cache.")
    parser.add_argument("--max_sequence_length", type=int, default=4096, help="Maximum number of tokens for model input")
    parser.add_argument("--prompt_template", type=str, default="LLaVA", help="Template name from PROMPT_DISTRIBUTIONS in examples.py")
    parser.add_argument("--disable_bbox_tree", action="store_true", help="Skip hierarchical bounding box analysis of images")
    parser.add_argument("--disable_filtering", action="store_true", help="Allow all generated samples without quality filtering (recommended when max_sample_count = 1)")
    parser.add_argument("--max_sample_count", type=int, default=10, help="Maximum number of language model samples per image")
    args = parser.parse_args()
    asyncio.run(main_async(args))