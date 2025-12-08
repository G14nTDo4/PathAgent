import os
import sys
import argparse

# --- 1. Parse Arguments FIRST (Before importing dependent libraries) ---
parser = argparse.ArgumentParser(description="Extract PLIP features for patches")

# Path to the library code (New Argument)
parser.add_argument("--plip_lib_path", type=str, required=True,
                    help="Path to the PLIP library directory (to be added to sys.path)")

# Other paths
parser.add_argument("--task_file", type=str, required=True, 
                    help="Path to the task text file containing folder names")
parser.add_argument("--img_root", type=str, required=True, 
                    help="Root directory containing patch images")
parser.add_argument("--output_dir", type=str, required=True, 
                    help="Directory to save the extracted features (.npy)")
parser.add_argument("--plip_ckpt", type=str, required=True, 
                    help="Path to the PLIP model checkpoint directory")

args = parser.parse_args()

# --- 2. Insert Library Path ---
if not os.path.exists(args.plip_lib_path):
    print(f"Error: PLIP library path does not exist: {args.plip_lib_path}")
    sys.exit(1)

sys.path.insert(0, args.plip_lib_path)

# --- 3. Import Dependent Libraries (Now safe to import PLIP) ---
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    from plip import PLIP
except ImportError as e:
    print(f"Error importing PLIP: {e}")
    print("Please check if the --plip_lib_path contains the 'plip.py' file.")
    sys.exit(1)

# --- Classes and Functions ---

class PatchDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return img, path

def custom_collate(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def run_processing(args):
    # --- Input Paths ---
    img_root = args.img_root
    output_feature_dir = args.output_dir
    plip_ckpt_path = args.plip_ckpt

    # --- Initialize Model ---
    if not os.path.exists(plip_ckpt_path):
        print(f"Error: Model checkpoint path does not exist: {plip_ckpt_path}")
        return

    print(f"Loading PLIP model from: {plip_ckpt_path}")
    plip = PLIP(plip_ckpt_path)

    # --- Read Task File ---
    if not os.path.exists(args.task_file):
        print(f"Error: Task file does not exist: {args.task_file}")
        return

    with open(args.task_file, "r") as f:
        task_ids = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Task file: {args.task_file}, {len(task_ids)} folders to process")
    print(f"Image Root: {img_root}")
    print(f"Output Dir: {output_feature_dir}")

    # --- Process Tasks ---
    for TARGET_LONG_ID in task_ids:
        print(f"\nStart processing folder: {TARGET_LONG_ID}")
        img_dir = os.path.join(img_root, TARGET_LONG_ID)
        output_dir = os.path.join(output_feature_dir, TARGET_LONG_ID)

        if not os.path.isdir(img_dir):
            print(f"{TARGET_LONG_ID} is not a directory, skipping")
            continue

        os.makedirs(output_dir, exist_ok=True)

        # Get all images
        image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
        if len(image_paths) == 0:
            print(f"{TARGET_LONG_ID} has no jpg files, skipping")
            continue

        saved_features = sorted(glob(os.path.join(output_dir, "*.npy")))

        if len(saved_features) == len(image_paths):
            print(f"{TARGET_LONG_ID} already completed ({len(saved_features)} features), skipping")
            continue
        else:
            print(f"{TARGET_LONG_ID} | Finished {len(saved_features)}/{len(image_paths)}")

        # Filter out already processed images
        done_set = set([os.path.splitext(os.path.basename(f))[0] for f in saved_features])
        remaining_paths = [p for p in image_paths if os.path.splitext(os.path.basename(p))[0] not in done_set]

        if not remaining_paths:
            continue

        dataset = PatchDataset(remaining_paths)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=32, collate_fn=custom_collate)

        total_saved_count = len(saved_features)
        for batch_imgs, batch_paths in tqdm(dataloader, desc=f"Encoding {TARGET_LONG_ID}", unit="batch"):
            batch_emb = plip.encode_images(batch_imgs, batch_size=len(batch_imgs))
            batch_emb = batch_emb / np.linalg.norm(batch_emb, ord=2, axis=-1, keepdims=True)

            for embedding, img_path in zip(batch_emb, batch_paths):
                base_name = os.path.basename(img_path)
                file_name_without_ext = os.path.splitext(base_name)[0]
                save_path = os.path.join(output_dir, f"{file_name_without_ext}.npy")
                np.save(save_path, embedding)
                total_saved_count += 1

        print(f"{TARGET_LONG_ID} processing complete, saved {total_saved_count} features")

if __name__ == "__main__":
    # Note: args are already parsed at the top level to allow dynamic imports
    run_processing(args)