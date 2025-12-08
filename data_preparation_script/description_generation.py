import os
import json
import torch
import argparse

from PIL import Image
from tqdm import tqdm

from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from transformers import TextStreamer

def save_result_grouped(json_path, folder_name, image_name, response):
    data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    if folder_name not in data:
        data[folder_name] = {}

    data[folder_name][image_name] = response

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main(args):
    print("===== Argument Configuration =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("================================\n")

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "pathology_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f'[WARNING] auto inferred conv_mode={conv_mode}, using {args.conv_mode}')
        conv_mode = args.conv_mode

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # Read slide list file
    with open(args.slide_list, "r") as f:
        selected_slides = [line.strip() for line in f if line.strip()]
    selected_slides = sorted(set(selected_slides))  # Deduplicate + Sort

    print(f"📂 Loaded {len(selected_slides)} slides to process from {args.slide_list}")

    root_dir = args.image_dir
    output_json = args.output_json

    # Read existing JSON to avoid duplicate processing
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Iterate through selected slides
    for folder in selected_slides:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Skipping non-existent folder: {folder_path}")
            continue

        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        img_files = sorted(img_files)
        total_patches = len(img_files)

        # Get count of processed patches
        done_patches = len(existing_data.get(folder, {}))

        # Check if slide is already finished
        if done_patches >= total_patches:
            print(f"⏩ Skipping finished slide: {folder}")
            continue
        else:
            print(f"▶️ Resuming/Processing slide: {folder} ({done_patches}/{total_patches} finished)")

        for img_file in tqdm(img_files, desc=f"Processing {folder}"):
            img_path = os.path.join(folder_path, img_file)
            image = Image.open(img_path).convert('RGB')
            image_tensor = process_images([image], image_processor, args)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)


            question = "Please describe the pathology features in this image."
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + question

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs

            # Save to JSON (write in real-time)
            save_result_grouped(output_json, folder, img_file, outputs)

            # Update existing_data in memory
            if folder not in existing_data:
                existing_data[folder] = {}
            existing_data[folder][img_file] = outputs

            # Clear conversation for the next image
            conv.messages = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing image patches")
    parser.add_argument("--output-json", type=str, default="image_descriptions.json", help="Path to output JSON file")
    parser.add_argument("--slide-list", type=str, required=True, help="Path to text file containing slide names to process")
    args = parser.parse_args()
    main(args)