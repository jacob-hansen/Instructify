# Instructify

Convert multi-modal datasets into instruction-tuning conversations using open source language models. Instructify processes images, captions, bounding boxes, and other metadata to create high-quality instruction-following data.

## Initial Setup

1. First, set up your Instructify Cache directory following the Visual Instruction Tuning instructions from [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning)

2. Create and setup the environment:
```bash
# Create and activate conda environment
conda create --name instruct python=3.11 -y
conda activate instruct

# Install dependencies 
pip install -r requirements.txt

# Set cache directory
export INSTRUCTIFY_CACHE=$(pwd)/data
```

## Dataset Preparation

Before generating instructions, you need to download and cache a dataset:

```bash
python setup.py \
  --datasets coco_captions \
  --download \
  --cache \
  --test-load-from-cache \
  --cache-name llava
```

## Instruction Generation

After dataset preparation, generate instructions:

```bash
python main.py \
  --run_id llava_replacement \
  --dataset_name llava.json \
  --model google/gemma-2-27b-it \
  --prompt_template LLaVA
```

## Processing Results

After generating instructions, process the results into LLaVA conversation format:

```bash
# View statistics about generated conversations
python process_results.py --run_id llava_replacement --count --detailed-count

# Export to LLaVA conversation format
python process_results.py --run_id llava_replacement --export output/llava_conversations.json
```

The exported JSON will be in LLaVA's standard conversation format:
```json
{
    "id": "unique_id",
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat do you see in this image?"
        },
        {
            "from": "gpt",
            "value": "I see..."
        }
    ]
}
```


## Command Line Arguments

Key arguments for controlling the pipeline:

```bash
--run_id             # Unique identifier for this processing run
--dataset_name       # Name of cached dataset JSON file
--model             # HuggingFace model ID (default: google/gemma-2-27b-it)
--num_workers       # Number of concurrent workers (default: 80)
--prompt_template   # Template for instruction generation (default: LLaVA)
--max_sample_count  # Max language model samples per image (default: 10)
```

Additional configuration:

```bash
--vllm_gpu_mem_fraction  # GPU memory allocation for LLM (0.0-0.85)
--output_path           # Directory for cache files
--max_sequence_length   # Maximum tokens for model input
--disable_bbox_tree     # Skip hierarchical bounding box analysis
--disable_filtering     # Skip quality filtering
```

Additional processing options:
```bash
--clean              # Remove empty result files
--remove             # Remove all result files
--detailed-count     # Show conversation statistics
--max-workers        # Number of parallel workers (default: 8)
```

## Troubleshooting

Common vLLM issues can be resolved by:

1. Decreasing GPU memory usage:
```bash
# Test model loading with reduced memory
vllm serve google/gemma-2-27b-it --tensor-parallel-size 2 --disable-custom-all-reduce
```

2. Adjusting the `vllm_gpu_mem_fraction` parameter:
```bash
python main.py \
  --run_id test_run \
  --dataset_name data.json \
  --vllm_gpu_mem_fraction 0.6
```

3. Known Warnings:
```
"You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset"
```
This warning can be safely ignored as it's related to Depth Anything V2, which has minimal impact on overall pipeline performance.