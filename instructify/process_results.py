import os
import json
import random
import argparse
from data_management import DatasetManager

def format_results(results, fill_in_blank_token="<fill-in-the-blank>", 
                  fill_in_blank_token_replacement="[blank]", 
                  image_dir=os.environ.get("INSTRUCTIFY_CACHE", "."), 
                  filter_out=set()):
    """Format results into LLaVA conversation format"""
    image_dir_representation = {}
    formatted_results = []
    
    for img_name, result_group in results.items():
        if len(result_group) == 0:
            continue
        for result in result_group[0]:
            if type(result) == str or result["prompt_type"] in filter_out:
                continue
                
            output = result["output"]
            conversation = {
                "id": f"instructify_processing_{random.randint(0, 1000000)}",
                "image": img_name,
                "conversations": []
            }
            
            for i, text in enumerate(output):
                conversation["conversations"].append({
                    "from": "human" if i % 2 == 0 else "gpt",
                    "value": text.replace(fill_in_blank_token, fill_in_blank_token_replacement)
                })

            conversation["conversations"][0]["value"] = "<image>\n" + conversation["conversations"][0]["value"]

            # Verify image exists
            image_sub_path = img_name[::-1].split("/", 1)[1][::-1]
            if image_sub_path not in image_dir_representation:
                image_dir_representation[image_sub_path] = set(os.listdir(os.path.join(image_dir, image_sub_path)))
            
            if img_name.split("/")[-1] not in image_dir_representation[image_sub_path]:
                print(f"Warning: Image {img_name} not found in directory")
                continue

            formatted_results.append(conversation)

    return formatted_results

def count_conversation_stats(results):
    """Count various statistics about the conversations"""
    conv_count = 0
    turn_count = 0
    image_count = 0
    
    for key in results:
        if len(results[key]) == 0:
            continue
        image_count += 1
        for turn in results[key][0]:
            if isinstance(turn, dict) and 'output' in turn:
                turn_count += len(turn['output'])
                conv_count += 1
    
    return {
        "image_count": image_count,
        "conv_count": conv_count,
        "turn_count": turn_count
    }

def main():
    parser = argparse.ArgumentParser(description="Manage dataset results")
    
    parser.add_argument("--run_id", type=str, required=True,
                       help="Target dataset/run ID to process")
    
    # Operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", action="store_true",
                      help="Count number of files only")
    group.add_argument("--clean", action="store_true",
                      help="Clean result files")
    group.add_argument("--remove", action="store_true",
                      help="Remove all result files")
    group.add_argument("--export", type=str,
                      help="Export formatted results to specified JSON path")
    
    # Additional options
    parser.add_argument("--detailed-count", action="store_true",
                       help="Show detailed conversation statistics")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Number of workers for parallel processing")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = DatasetManager(max_workers=args.max_workers)
    
    if args.count:
        count = manager.count_results(args.run_id)
        print(f"Total files: {count}")
        
        if args.detailed_count:
            results = manager.collect_results(args.run_id)
            stats = count_conversation_stats(results)
            print(f"Images processed: {stats['image_count']}")
            print(f"Total conversations: {stats['conv_count']}")
            print(f"Total turns: {stats['turn_count']}")
    
    elif args.clean:
        manager.clean(args.run_id, empty_only=True)
        print(f"Cleaned empty results for {args.run_id}")
    
    elif args.remove:
        # get input to double check
        print(f"Are you sure you want to remove all results for {args.run_id}? (y/n)")
        confirm = input()
        if confirm.lower() != 'y':
            print("Operation cancelled")
            return

        manager.clean(args.run_id, empty_only=False)
        print(f"Removed all results for {args.run_id}")
    
    elif args.export:
        # Validate json path
        export_path = args.export
        if not export_path.endswith('.json'):
            export_path += '.json'
            
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        
        # Collect and format results
        results = manager.collect_results(args.run_id)
        formatted_results = format_results(results)
        
        # Save to JSON
        with open(export_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        print(f"Results exported to {export_path}")

if __name__ == "__main__":
    main()