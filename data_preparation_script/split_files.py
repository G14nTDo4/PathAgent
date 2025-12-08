import os
import math
import argparse

def split_all_slides(image_dir, save_dir, num_splits):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(image_dir):
        print(f"Error: Input directory '{image_dir}' does not exist.")
        return

    all_slides = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    all_slides = sorted(all_slides)
    total_slides = len(all_slides)
    print(f"✅ Found {total_slides} slides in total.")

    if total_slides == 0:
        print("No slides found. Please check the directory path.")
        return


    split_size = (total_slides + num_splits - 1) // num_splits
    splits = [all_slides[i * split_size : (i + 1) * split_size] for i in range(num_splits)]
    for i, split in enumerate(splits):
        save_path = os.path.join(save_dir, f"slides_part{i+1}.txt")
        with open(save_path, "w") as f:
            for slide in split:
                f.write(slide + "\n")
        print(f"Saved {len(split)} slides to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split slide folders into multiple text files.")
    
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Path to the directory containing slide folders")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Path to the directory where txt files will be saved")
    parser.add_argument("--num_splits", type=int, default=4, 
                        help="Number of parts to split the list into (default: 4)")
    args = parser.parse_args()

    print("="*30)
    print(f"Source Dir:  {args.image_dir}")
    print(f"Save Dir:    {args.save_dir}")
    print(f"Num Splits:  {args.num_splits}")
    print("="*30)

    split_all_slides(args.image_dir, args.save_dir, args.num_splits)