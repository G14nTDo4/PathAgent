import os
import gc
import sys
import json
import torch
import warnings
import argparse
import numpy as np
from copy import deepcopy

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="WSI-VQA Inference Pipeline")

    # --- Library Paths ---
    parser.add_argument("--plip_lib_path", type=str, required=True,
                        help="Path to the PLIP library directory")

    # --- Model Checkpoints ---
    parser.add_argument("--qwen_ckpt", type=str, required=True, 
                        help="Path to Qwen checkpoint")
    parser.add_argument("--plip_ckpt", type=str, required=True, 
                        help="Path to PLIP checkpoint")
    parser.add_argument("--patho_r1_ckpt", type=str, required=True, 
                        help="Path to Patho-R1 checkpoint")

    # --- Data Files ---
    parser.add_argument("--descriptions_file", type=str, required=True, 
                        help="Path to patch descriptions JSON file")
    parser.add_argument("--questions_file", type=str, required=True, 
                        help="Path to questions/VQA dataset JSON file")
    parser.add_argument("--feature_dir", type=str, required=True, 
                        help="Directory containing image features")
    parser.add_argument("--patch_root", type=str, required=True, 
                        help="Root directory for image patches")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Directory to save results")

    # --- Settings ---
    parser.add_argument("--dataset_name", type=str, default="wsi_vqa", 
                        help="Name of the dataset (e.g., wsi_vqa, slidechat)")

    return parser.parse_args()

# --- 1. Parse Arguments First ---
args = parse_args()

# --- 2. Dynamic Path Insertion ---
if not os.path.exists(args.plip_lib_path):
    raise FileNotFoundError(f"PLIP library path not found: {args.plip_lib_path}")

sys.path.insert(0, args.plip_lib_path)
from plip import PLIP
from data_processing.utils import * 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from models.inference import evaluate_with_llm_chain, slide_llm_answer, patho_r1_describe, summarize_patches_in_chunks

def main():
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*40)
    print(f"PLIP Lib:   {args.plip_lib_path}")
    print(f"Model Qwen: {args.qwen_ckpt}")
    print(f"Model PLIP: {args.plip_ckpt}")
    print(f"Model PR1:  {args.patho_r1_ckpt}")
    print(f"Dataset:    {args.dataset_name}")
    print(f"Results to: {args.save_dir}")
    print("="*40)

    # Load VQA Pairs
    print(f"Loading VQA pairs from {args.questions_file}...")
    pairs = load_all_vqa_pairs(
        args.questions_file, 
        dataset_name=args.dataset_name, 
        image_dir=args.patch_root
    )
    print(f"Successfully loaded {len(pairs)} VQA pairs.")

    initial_sample_ratio = 0.10
    replenish_ratio = 0.05
    random_seed = 128
    zoom_level_val = 5

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_ckpt, 
        torch_dtype="auto", 
        device_map="auto"
    )
    print("Qwen model loaded successfully.")

    # PLIP
    plip = PLIP(args.plip_ckpt)
    print("PLIP model loaded successfully.")

    # Patho-R1
    patho_r1_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.patho_r1_ckpt,
        torch_dtype="auto",
        device_map="auto"
    )
    patho_r1_processor = AutoProcessor.from_pretrained(args.patho_r1_ckpt)
    print("Patho-R1_7B model loaded successfully.")

    # --- Data Preparation ---
    finished_cases = set()
    if os.path.exists(args.save_dir):
        for f in os.listdir(args.save_dir):
            if f.endswith(".json"):
                finished_cases.add(f.replace(".json", ""))

    with open(args.descriptions_file, "r", encoding="utf-8") as f:
        all_descriptions = json.load(f)
    all_keys = list(all_descriptions.keys())
    print(f"Pre-loaded descriptions file, containing {len(all_keys)} keys.")

    all_case_results = []
    max_attempts = 5

    for idx, pair in enumerate(pairs):
        TARGET_LONG_ID = pair["long_id"]
        
        matching_keys = [k for k in all_keys if k.startswith(TARGET_LONG_ID)]
        if len(matching_keys) == 1:
            TARGET_LONG_ID = matching_keys[0]
        elif len(matching_keys) > 1:
            print(f"Warning: Found multiple keys starting with '{TARGET_LONG_ID}', using the first one: {matching_keys[0]}")
            TARGET_LONG_ID = matching_keys[0]
        else:
            print(f"Target ID '{TARGET_LONG_ID}' not found in description file, keeping as is.")

        question_text = pair["question"]
        question_choices = pair["choices"]
        correct_answer = pair["answer"]

        # Assuming make_unique_id is imported from data_processing.utils
        unique_id = make_unique_id(TARGET_LONG_ID, question_text)

        save_path = os.path.join(args.save_dir, f"{unique_id}.json")

        # Resume from breakpoint
        if unique_id in finished_cases:
            print(f"Skipping Case {idx+1}/{len(pairs)}: {unique_id} (Already completed)")
            continue

        print(f"\nStarting Case {idx+1}/{len(pairs)}: {unique_id}")

        print("="*80)
        print(f"Question: {question_text}")
        print(f"Choices: {question_choices}")
        print(f"Ground Truth: {correct_answer}")

        # --- Step 1: Extract Data ---
        descriptions_dict = get_specific_case_descriptions(args.descriptions_file, TARGET_LONG_ID)
        descriptions_dict = deepcopy(descriptions_dict)
        print(f"Extracted {len(descriptions_dict)} patch descriptions.")

        # --- Step 2: Preload Features ---
        feature_cache = {}
        all_patch_names = list(descriptions_dict.keys())
        for patch_name in all_patch_names:
            # Assuming filename format aligns with your logic
            npy_path = os.path.join(args.feature_dir, TARGET_LONG_ID, f"{patch_name.split('.')[0]}.npy")
            if os.path.exists(npy_path):
                feature_cache[patch_name] = np.load(npy_path)
        print(f"Successfully cached {len(feature_cache)} / {len(all_patch_names)} patch features")

        # --- Step 3: Initialization ---
        q_emb = plip.encode_text([question_text], batch_size=1)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=-1, keepdims=True)
        available_patches = [p for p in all_patch_names if p in feature_cache]

        if not available_patches:
            print("No available patches with features found. Skipping case.")
            continue

        feats = np.stack([feature_cache[p] for p in available_patches], axis=0)
        # feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        sims_all = (feats @ q_emb.T).squeeze()

        K = max(1, int(len(all_patch_names) * initial_sample_ratio))
        topk_idx = np.argsort(sims_all)[-K:][::-1]
        initial_sample_names = [available_patches[i] for i in topk_idx]

        accumulated_patch_names = list(initial_sample_names)
        patches_to_evaluate_names = list(initial_sample_names)
        remaining_patch_names = [p for p in all_patch_names if p not in accumulated_patch_names]

        print(f"Initial sample size: {len(initial_sample_names)}; Remaining: {len(remaining_patch_names)}")

        # --- Step 4: Loop Evaluation ---
        attempt = 0
        final_answer = None
        all_results = []

        while attempt < max_attempts:
            attempt += 1
            print(f"\nLoop attempt {attempt}... Evaluating {len(patches_to_evaluate_names)} patches this round.")
            if not patches_to_evaluate_names:
                print("No new patches available for evaluation this round, stopping.")
                break

            # ------------- Step A: Generate question-specific descriptions for top N patches -------------
            num_top_patches_to_describe = 5
            num_top_patches_to_describe = min(num_top_patches_to_describe, len(patches_to_evaluate_names))
            for i in range(num_top_patches_to_describe):
                patch_name = patches_to_evaluate_names[i]
                patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
                patch_img = load_image(patch_path)

                x, y = extract_coords_from_name(patch_name)
                question_specific_desc = patho_r1_describe(
                    patch_img,
                    question=question_text,
                    patho_r1_processor=patho_r1_processor,
                    patho_r1_model=patho_r1_model,
                    coords=(x, y),
                    magnification=zoom_level_val,
                    choices=question_choices
                )
                
                orig = descriptions_dict.get(patch_name, "")
                descriptions_dict[patch_name] = (orig + " " + question_specific_desc).strip()
                print(f"Appended question-specific description for {patch_name} (Total length {len(descriptions_dict[patch_name])})")


            # ------------- Step B: Construct meta-descriptions and submit to patch-evaluator -------------
            current_items_to_evaluate = [(name, descriptions_dict[name]) for name in patches_to_evaluate_names]
            desc_for_evaluation = build_descriptions_with_meta(current_items_to_evaluate, mag_level=zoom_level_val, include_header=True, include_coords=True)

            evaluation_result = evaluate_with_llm_chain(model, tokenizer, desc_for_evaluation, question_text, question_choices)
            can_now = str(evaluation_result.get("sufficient", "")).strip().lower() == "yes"
            can_zoom = str(evaluation_result.get("zoom_recommendation", "")).strip().lower() == "yes"

            # ------------- Handle Branches -------------
            if can_now:
                print("\npatch-llm judgment: Can answer now.")
                final_desc_text = summarize_patches_in_chunks(
                    model, tokenizer,
                    descriptions_dict, accumulated_patch_names,
                    question_text=question_text,
                    chunk_size=10, threshold=50,
                    magnification=zoom_level_val
                )
                final_answer = slide_llm_answer(
                    model, tokenizer, 
                    final_desc_text, 
                    question_text, 
                    question_choices, 
                    magnification=zoom_level_val, 
                    case_name=unique_id, 
                ) 

                all_results = [{
                    "attempt": attempt,
                    "mode": "can_answer_now",
                    "evaluated_patches_this_round": patches_to_evaluate_names,
                    "total_accumulated_patches": len(accumulated_patch_names),
                    "evaluation_result": evaluation_result,
                    "answer": final_answer["answer"],
                    "explanation": final_answer["explanation"]
                }]
                break

            elif can_zoom:

                final_desc_text = summarize_patches_in_chunks(
                    model, tokenizer,
                    descriptions_dict, accumulated_patch_names,
                    question_text=question_text,
                    chunk_size=10, threshold=50,
                    magnification=zoom_level_val
                )
                zoom_reason = evaluation_result.get("zoom_reason", "").strip()
                print(f"\npatch-evaluator judgment: Need zoom. Reason for zoom is {zoom_reason}")
                # parse zoom level from evaluator
                zoom_level_raw = evaluation_result.get("zoom_level", zoom_level_val)
                try:
                    zoom_level_val = int(zoom_level_raw)
                except (ValueError, TypeError):
                    zoom_level_val = zoom_level_val  # keep current

                print(f"  -> Zoom level: {zoom_level_val}x")

                patch_feats = []
                patch_names_valid = []
                for p in patches_to_evaluate_names:
                    if p in feature_cache:
                        patch_feats.append(feature_cache[p])
                        patch_names_valid.append(p)
                if not patch_feats:
                    print("No valid patch features, skipping zoom process.")
                else:
                    patch_feats = np.stack(patch_feats, axis=0)
                    # patch_feats = patch_feats / np.linalg.norm(patch_feats, axis=-1, keepdims=True)
                    text_emb = plip.encode_text([question_text], batch_size=1)
                    text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
                    sims = np.dot(patch_feats, text_emb.T).squeeze()

                    top2_idx = np.argsort(sims)[-2:][::-1]
                    top2_names = [patch_names_valid[i] for i in top2_idx]
                    print(f"  -> Selected Top2 patches: {top2_names}")

                    # split top2 patches into sub-patches at zoom_level_val
                    candidate_images = []
                    candidate_names = []
                    for patch_name in top2_names:
                        patch_path = get_patch_fullpath(args.patch_root, TARGET_LONG_ID, patch_name)
                        sub_patches = split_patch_for_zoom(patch_path, zoom_level_val)
                        for sub_img, (x, y) in sub_patches:
                            candidate_images.append(sub_img)
                            candidate_names.append(f"{x}_{y}")

                    if not candidate_images:
                        print("No sub-patches generated, skipping zoom process.")
                    else:
                        image_embs = plip.encode_images(candidate_images, batch_size=4)
                        image_embs = image_embs / np.linalg.norm(image_embs, axis=-1, keepdims=True)
                        sims_sub = np.dot(image_embs, text_emb.T).squeeze()
                        best_idx = int(np.argmax(sims_sub))
                        best_img = candidate_images[best_idx]
                        best_name = candidate_names[best_idx]
                        print(f"Selected zoomed sub-patch: {best_name}")

                        x, y = best_name.split("_")

                        zoomed_patch_desc = patho_r1_describe(
                            best_img,
                            question=question_text,
                            patho_r1_processor=patho_r1_processor,
                            patho_r1_model=patho_r1_model,
                            coords=(x, y),
                            magnification=zoom_level_val,
                            choices=question_choices
                        )

                        extra_note = (
                            f"\n\n[NOTE] A key sub-patch was identified after zooming.\n"
                            f"Patch=({x},{y}), Magnification={zoom_level_val}x\n"
                            f"Detail: {zoomed_patch_desc}"
                        )
                        final_desc_text = final_desc_text + extra_note

                        final_answer = slide_llm_answer(
                            model, tokenizer,
                            final_desc_text,
                            question_text, question_choices,
                            magnification=zoom_level_val,
                            case_name=unique_id,
                        )

                        all_results = [{
                            "attempt": attempt,
                            "mode": "zoom_then_select",
                            "evaluated_patches_this_round": patches_to_evaluate_names,
                            "total_accumulated_patches": len(accumulated_patch_names),
                            "evaluation_result": evaluation_result,
                            "selected_zoom_patch": best_name,
                            "zoom_patch_desc": zoomed_patch_desc,
                            "answer": final_answer["answer"],
                            "explanation": final_answer["explanation"]
                        }]
                        break

            else:
                # cannot answer and cannot zoom -> find additional patches relevant to missing_info
                missing_info = evaluation_result.get("missing_info", "pathology details").strip()
                print(f"patch-evaluator identifies missing info: '{missing_info}' -> Searching for new patches.")

                if not remaining_patch_names:
                    print("No more remaining patches available, loop terminated.")
                    break

                valid_remaining_names = [name for name in remaining_patch_names if name in feature_cache]
                if not valid_remaining_names:
                    print("No valid feature cache for remaining patches, loop terminated.")
                    break

                remaining_embs = np.stack([feature_cache[name] for name in valid_remaining_names], axis=0)
                text_emb = plip.encode_text([missing_info], batch_size=1)

                # remaining_embs = remaining_embs / np.linalg.norm(remaining_embs, axis=-1, keepdims=True)
                text_emb = text_emb / np.linalg.norm(text_emb, axis=-1, keepdims=True)
                sims = np.dot(remaining_embs, text_emb.T).squeeze()

                num_to_add = max(1, int(len(all_patch_names) * replenish_ratio))
                best_indices = np.argsort(sims)[::-1][:num_to_add]

                newly_selected_patches = [valid_remaining_names[i] for i in best_indices]
                print(f"Selected {len(newly_selected_patches)} new relevant patches.")

                patches_to_evaluate_names = newly_selected_patches
                accumulated_patch_names.extend(newly_selected_patches)
                remaining_patch_names = [p for p in remaining_patch_names if p not in newly_selected_patches]

                print(f"   Total accumulated patches: {len(accumulated_patch_names)}, Remaining available: {len(remaining_patch_names)}")

        if final_answer is None:
            print("Answer not found, entering fallback...")
            final_desc_text = summarize_patches_in_chunks(
                model, tokenizer,
                descriptions_dict, accumulated_patch_names,
                question_text=question_text,
                chunk_size=10, threshold=50,
                magnification=zoom_level_val
            )
            final_answer = slide_llm_answer(
                model, tokenizer, 
                final_desc_text, 
                question_text, 
                question_choices, 
                magnification=zoom_level_val, 
                case_name=unique_id, 
            )

            all_results.append({
                "attempt": "final_fallback",
                "mode": "fallback_final_attempt",
                "total_accumulated_patches": len(accumulated_patch_names),
                "answer": final_answer["answer"],
                "explanation": final_answer["explanation"]
            })
        print('single_case_result:', final_answer["answer"])
        print('ground_truth:', correct_answer)
        
        # --- Save Single Case Result ---
        case_result = {
            "long_id": TARGET_LONG_ID,
            "question": question_text,
            "choices": question_choices,
            "ground_truth": correct_answer,
            "pred_answer": final_answer["answer"],
            "explanation": final_answer["explanation"],
            "process": all_results
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(case_result, f, indent=2, ensure_ascii=False)

        all_case_results.append(case_result)

        gc.collect()
        torch.cuda.empty_cache()

    # --- Final Summary Output ---
    print("\n\n========== All Results ==========")
    print(json.dumps(all_case_results, indent=2, ensure_ascii=False))
    
if __name__ == "__main__":
    main()