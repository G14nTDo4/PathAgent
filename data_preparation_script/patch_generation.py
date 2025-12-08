import os
import h5py
import argparse
import openslide

from PIL import Image
from pathlib import Path

def extract_patches_from_h5(h5_path, slide_path, save_dir, patch_size=4096, level=0):
    """
    Extract patches from a single WSI based on coordinates in an h5 file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Read coordinates
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
    
    slide = openslide.OpenSlide(slide_path)
    
    for (x, y) in coords:
        try:
            # OpenSlide requires (x, y) at level 0 reference
            patch = slide.read_region((int(x), int(y)), level, (patch_size, patch_size)).convert('RGB')
            patch_name = f"{int(x)}_{int(y)}.jpg"
            patch.save(os.path.join(save_dir, patch_name), "JPEG")
        except Exception as e:
            print(f"Error extracting patch at {x},{y}: {e}")
            
    slide.close()

def batch_extract(h5_dir, slide_dir, output_root, patch_size=4096, level=0):
    """
    Batch process all h5 files in a directory.
    """
    h5_files = sorted(Path(h5_dir).glob("*.h5"))
    total_files = len(h5_files)
    
    print(f"Found {total_files} h5 files to process.")

    for idx, h5_file in enumerate(h5_files, start=1):
        case_id = h5_file.stem
        save_dir = Path(output_root) / case_id

        # Check total coordinates count for resume logic
        with h5py.File(h5_file, 'r') as f:
            coords_count = f['coords'].shape[0]

        # Resume / Skip check
        if save_dir.exists():
            existing_files = list(save_dir.glob("*.jpg"))
            if len(existing_files) == coords_count:
                print(f"[Skipping {idx}/{total_files}] {case_id} already finished ({coords_count} patches)")
                continue

        print(f"[{idx}/{total_files}] Processing: {h5_file.name}")
        
        # Assuming slides end with .svs. Modify extension here if needed (e.g., .tiff, .ndpi)
        svs_path = Path(slide_dir) / f"{case_id}.svs"
        
        if not svs_path.exists():
            print(f"[Skipping] Slide file not found: {svs_path}")
            continue

        extract_patches_from_h5(
            h5_path=str(h5_file),
            slide_path=str(svs_path),
            save_dir=str(save_dir),
            patch_size=patch_size,
            level=level
        )

    print(f"[Done] Processed {total_files} h5 files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract patches from WSI using H5 coordinates')
    
    # Path arguments
    parser.add_argument('--h5_dir', type=str, required=True, 
                        help='Directory containing .h5 coordinate files')
    parser.add_argument('--slide_dir', type=str, required=True, 
                        help='Directory containing raw WSI slides (.svs)')
    parser.add_argument('--output_root', type=str, required=True, 
                        help='Root directory to save extracted patches')
    
    # Parameter arguments
    parser.add_argument('--patch_size', type=int, default=4096, 
                        help='Size of the patches to extract (default: 4096)')
    parser.add_argument('--level', type=int, default=0, 
                        help='Mag level to extract from (default: 0)')

    args = parser.parse_args()

    # Print configuration
    print("-" * 30)
    print(f"H5 Dir:      {args.h5_dir}")
    print(f"Slide Dir:   {args.slide_dir}")
    print(f"Output Root: {args.output_root}")
    print(f"Patch Size:  {args.patch_size}")
    print(f"Level:       {args.level}")
    print("-" * 30)

    batch_extract(
        h5_dir=args.h5_dir,
        slide_dir=args.slide_dir,
        output_root=args.output_root,
        patch_size=args.patch_size,
        level=args.level
    )