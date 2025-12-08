import os
import json
import argparse
from glob import glob

def merge_json_files(input_dir, output_file):
    # Search for all .json files in the input directory
    json_files = sorted(glob(os.path.join(input_dir, "*.json")))
    
    # Filter out the output file itself if it exists in the same directory
    # to prevent trying to merge the output into itself
    json_files = [f for f in json_files if os.path.abspath(f) != os.path.abspath(output_file)]

    if not json_files:
        print(f"No JSON files found in directory: {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to merge.")
    
    merged_data = {}

    for jf in json_files:
        print(f"Processing: {os.path.basename(jf)}")
        
        if not os.path.exists(jf):
            print(f"Skipping non-existent file: {jf}")
            continue
            
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Failed to parse {jf}, skipping.")
                continue

        # Merge Logic: If slide exists, update its patches; otherwise create new entry
        for slide, patches in data.items():
            if slide not in merged_data:
                merged_data[slide] = {}
            merged_data[slide].update(patches)

    print(f"Successfully merged {len(json_files)} files.")
    print(f"Total unique slides: {len(merged_data)}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Merged data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON files from a directory into one.")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing the JSON part files")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path where the merged JSON will be saved")

    args = parser.parse_args()
    merge_json_files(args.input_dir, args.output_file)