import os
import re
import json
import torch

from copy import deepcopy
from qwen_vl_utils import process_vision_info
from data_processing.utils import extract_coords_from_name, build_descriptions_with_meta

def _call_llm_return_json_simple(model, tokenizer, messages, max_new_tokens=256, retries=1):
    """
    Call the qwen LLM and parse JSON.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    def extract_json_block(s: str):
        """Extract the first complete { ... } block from the generated text (based on brace counting)."""
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    for attempt in range(retries + 1):
        with torch.no_grad():
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Attempt to parse directly
        try:
            parsed = json.loads(full_output)
            return full_output, parsed
        except Exception:
            # Attempt to extract the first JSON block
            json_block = extract_json_block(full_output)
            if json_block:
                try:
                    parsed = json.loads(json_block)
                    return full_output, parsed
                except Exception:
                    pass

        # If failed, add a reminder prompt and try again
        if attempt < retries:
            messages_retry = deepcopy(messages)
            messages_retry.append({
                "role": "system",
                "content": "Reminder: Respond with only a single valid JSON object containing the requested keys."
            })
            text = tokenizer.apply_chat_template(
                messages_retry,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            continue

        return full_output, None

    return None, None

def evaluate_with_llm_chain(model, tokenizer, description, question, choices=None,
                               max_new_tokens_a=512, max_new_tokens_b=256, max_new_tokens_c=256,
                               retries=1):
    """
    Three-step logic chain (simplified):
      Step A: Generate answer + thinking_steps
      Step B: Judge if sufficient (Yes / No)
      Step C: If insufficient, analyze missing info and zoom strategy
    """
    # --- Step A ---
    system_a = (
        "You are an expert AI pathology assistant.\n"
        "Task: Based on the patch descriptions, try to answer the question step-by-step.\n"
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "answer": "the final predicted answer (string)",\n'
        '  "thinking_steps": "your detailed reasoning, step-by-step (string)"\n'
        "}"
    )

    choices_text = f"\nChoices: {choices}" if choices else ""
    user_a = f"--- Patch Descriptions ---\n{description}\n--- End ---\nQuestion: {question}{choices_text}\nNow output the JSON with 'answer' and 'thinking_steps'."

    messages_a = [
        {"role": "system", "content": system_a},
        {"role": "user", "content": user_a},
    ]
    full_a, parsed_a = _call_llm_return_json_simple(model, tokenizer, messages_a,
                                                   max_new_tokens=max_new_tokens_a, retries=retries)

    if parsed_a is None:
        parsed_a = {
            "answer": "Uncertain",
            "thinking_steps": full_a or ""
        }

    parsed_a.setdefault("answer", "Uncertain")
    parsed_a.setdefault("thinking_steps", "")

    # --- Step B ---
    system_b = (
        "You are an expert AI pathology assistant.\n"
        "Task: Judge whether the current patch descriptions are sufficient to confidently support the answer.\n"
        "Output ONLY a JSON object like:\n"
        '{"sufficient": "Yes" or "No" }'
    )
    user_b = (
        f"Descriptions:\n{description}\n\n"
        f"Question: {question}{choices_text}\n\n"
        f"Previous answer and reasoning:\n{json.dumps(parsed_a, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )
    messages_b = [{"role": "system", "content": system_b}, {"role": "user", "content": user_b}]
    full_b, parsed_b = _call_llm_return_json_simple(model, tokenizer, messages_b,
                                                   max_new_tokens=max_new_tokens_b, retries=retries)

    if parsed_b is None:
        parsed_b = {"sufficient": "Uncertain"}

    suff = parsed_b.get("sufficient", "").strip().lower()

    # --- If sufficient == "yes", return immediately ---
    if suff == "yes":
        return {
            "answer": parsed_a.get("answer"),
            "thinking_steps": parsed_a.get("thinking_steps"),
            "sufficient": parsed_b.get("sufficient", ""),
            "raw_texts": {
                "step_a_raw": full_a,
                "step_b_raw": full_b,
                "step_c_raw": None
            }
        }

    # --- Step C ---
    system_c = (
        "You are an expert AI pathology assistant.\n"
        "Task: If current data is insufficient, specify what visual evidence is missing, "
        "and whether zooming in could help obtain that evidence.\n"
        "Output ONLY a JSON object like:\n"
        "{\n"
        '  "missing_info": "short noun phrase",\n'
        '  "zoom_recommendation": "Yes" or "No",\n'
        '  "recommended_zoom_level": "None" or an integer like 10 or 20 or 40,\n'
        '  "zoom_reason": "brief reason why zooming helps"\n'
        "}"
    )

    user_c = (
        f"Descriptions:\n{description}\n\n"
        f"Question: {question}{choices_text}\n\n"
        f"Previous answer: {json.dumps(parsed_a, ensure_ascii=False)}\n"
        f"Sufficiency judgement: {json.dumps(parsed_b, ensure_ascii=False)}\n\n"
        "Now provide the JSON for missing info and zoom recommendation."
    )
    messages_c = [{"role": "system", "content": system_c}, {"role": "user", "content": user_c}]
    full_c, parsed_c = _call_llm_return_json_simple(model, tokenizer, messages_c,
                                                   max_new_tokens=max_new_tokens_c, retries=retries)

    if parsed_c is None:
        parsed_c = {
            "missing_info": "Uncertain",
            "zoom_recommendation": "Uncertain",
            "recommended_zoom_level": "Uncertain",
            "zoom_reason": full_c or ""
        }

    return {
        "answer": parsed_a.get("answer"),
        "thinking_steps": parsed_a.get("thinking_steps"),
        "sufficient": parsed_b.get("sufficient"),
        "missing_info": parsed_c.get("missing_info"),
        "zoom_recommendation": parsed_c.get("zoom_recommendation"),
        "zoom_level": parsed_c.get("recommended_zoom_level", 5),
        "zoom_reason": parsed_c.get("zoom_reason"),
        "raw_texts": {
            "step_a_raw": full_a,
            "step_b_raw": full_b,
            "step_c_raw": full_c
        }
    }

def slide_llm_answer(
    model,
    tokenizer,
    descriptions_text,
    question,
    choices=None,
    magnification=None,
    case_name=None,
):
    """
    Slide-level LLM: Generates the final answer based on multiple patch descriptions.
    Optimized the prompt to focus the model on specific slide-level results rather than conceptual explanations.
    Automatically repairs JSON format errors in the model output.
    """
    system_prompt = (
        "You are an expert slide-level pathology assistant. "
        "You will be given a question and detailed patch-level descriptions of a pathology slide. "
        "Your task is to infer the specific **slide-level diagnostic result or quantitative value** "
        "based on the provided evidence — not to define or explain the medical term itself. "
        "The answer should directly reflect the information observable in the slide, such as biomarker expression level, "
        "presence or absence of features, or a numeric measurement.\n\n"
        "Rules:\n"
        "1. When the question asks about a biomarker (e.g., HER2, progesterone receptor, Ki-67), "
        "output the **observed status or score** (e.g., 'positive', 'negative', '2+', 'high expression'), not the definition.\n"
        "2. When the question asks about survival time or other quantitative results, "
        "output only the numerical value for the answer key in the response JSON.\n"
        "3. Always keep the 'answer' short — short phrase, or one number.\n"
        "4. Always include a brief reasoning in 'explanation', summarizing how the evidence supports the answer.\n"
        "5. **If choices are provided, your 'answer' must be exactly one of the given options. "
        "Never generate an answer outside the provided choices.**\n\n"
        "Respond strictly in JSON format with keys: 'answer' and 'explanation'."
    )

    if magnification is not None:
        system_prompt = f"[Slide-level Context | Magnification={magnification}x]\n" + system_prompt

    choices_text = f"\nChoices: {choices}" if choices else ""

    user_prompt = (
        f"Question: {question}{choices_text}\n\n"
        "Now, based on the following patch-level descriptions of the slide, "
        "determine the **slide-level result** that directly answers the question.\n\n"
        f"--- Patch Descriptions ---\n{descriptions_text}\n--- End of Descriptions ---\n\n"
        "Answer in JSON format:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    with torch.no_grad():
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.15,
            do_sample=False
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    raw_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # ------------------- JSON Extraction and Repair Logic -------------------
    parsed = None
    json_candidates = re.findall(r"\{.*?\}", raw_answer, flags=re.DOTALL)

    for candidate in reversed(json_candidates):  # Parse the last one first
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            continue

    if parsed is None:
        print(f"JSON parsing failed ({case_name if case_name else 'unknown_case'}), raw output:\n{raw_answer}\n")

        # fallback: Extract the first word as the answer
        answer = {
            "answer": raw_answer.split()[0] if raw_answer else "Unknown",
            "explanation": "Model did not return valid JSON."
        }
    else:
        answer_text = parsed.get("answer", "").strip()
        if not answer_text:
            answer_text = "Unknown"

        explanation_text = parsed.get("explanation", "").strip()
        if not explanation_text:
            explanation_text = "No explanation provided."

        answer = {
            "answer": answer_text,
            "explanation": explanation_text
        }

    del model_inputs, generated_ids, output_ids
    torch.cuda.empty_cache()

    return answer

def patho_r1_describe(image, question=None, 
                    patho_r1_processor=None, patho_r1_model=None,
                    coords=None, magnification=None, max_new_tokens=1024, choices=None, missing_info=None):

    meta_lines = []
    if coords is not None:
        meta_lines.append(f"Patch coordinates: ({coords[0]},{coords[1]})")
    if magnification is not None:
        meta_lines.append(f"Magnification: {magnification}x")
    meta_text = ""
    if meta_lines:
        meta_text = "[IMAGE META] " + " | ".join(meta_lines) + "\n\n"


    if question is None:
        prompt_body = "Please describe the pathology features in this image."
    else:
        prompt_body = (
            f"Question: {question}\n\n"
            "Answer the question and list the pathological features visible in the image that support your answer."
        )

    if choices is not None:
        prompt_body += f"\nChoices: {choices}"
    if missing_info is not None:
        prompt_body += f"\nMissing information: {missing_info}"

    full_text = meta_text + prompt_body

    # === Construct system prompt ===
    system_prompt = (
        "A conversation between a curious user and an AI medical assistant specialized in pathology image analysis. "
        "The assistant can interpret pathology images, describe observed features, and provide possible explanations based on medical knowledge, "
        "but will never give a definitive diagnosis or prescribe treatment. "
        "The assistant must always maintain a polite, clear, and professional tone. "
        "All answers should be supported by established, reliable medical sources. "
        "The assistant should carefully consider visual details in pathology images, such as cell morphology, staining patterns, and tissue architecture. "
        "If choices are given, the answer must be given from the choices."
        "If Missing information is given, you need to focus on this part of the image."
    )

    # === Construct message input ===
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_text},
            ],
        },
    ]

    # === Construct model input ===
    text = patho_r1_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = patho_r1_processor(
        text=[text],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(patho_r1_model.device)

    # === Inference generation ===
    with torch.no_grad():
        generated_ids = patho_r1_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = patho_r1_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return output_text

def summarize_patches_in_chunks(
    model, tokenizer, descriptions_dict, patch_names,
    question_text=None, chunk_size=5, threshold=50, magnification=None
):
    """
    If the number of patches exceeds the threshold, summarize descriptions in chunks of `chunk_size`.
    The summary for each chunk will explicitly list the included patch coordinates and conduct a guided summary based on the question.
    """
    if len(patch_names) <= threshold:
        # Does not exceed threshold, concatenate original descriptions directly
        items = [(name, descriptions_dict[name]) for name in patch_names]
        return build_descriptions_with_meta(
            items, mag_level=magnification, include_header=True, include_coords=True
        )

    print(f"Patch count {len(patch_names)} exceeds {threshold}, performing chunked summarization...")
    summaries = []
    for i in range(0, len(patch_names), chunk_size):
        chunk_names = patch_names[i:i+chunk_size]
        items = [(name, descriptions_dict[name]) for name in chunk_names]

        # Extract coordinate list
        coords_list = []
        for name, _ in items:
            x, y = extract_coords_from_name(name)
            coords_list.append(f"({x},{y})" if x is not None and y is not None else "(unknown)")
        coords_str = ", ".join(coords_list)

        # Concatenate patch descriptions (with coordinates)
        chunk_text = build_descriptions_with_meta(
            items, mag_level=magnification, include_header=False, include_coords=True
        )

        # === Construct Prompt ===
        system_prompt = (
            "You are an expert pathology assistant. "
            "You will be given multiple patch-level descriptions of histopathology images. "
            "Your task is to summarize the key pathological features across these patches. "
            "The summary must be concise yet informative, highlighting significant morphological patterns. "
            "Additionally, focus on information that could help answer the following question "
            "about the slide, emphasizing details relevant to the diagnostic or interpretive context."
        )

        if magnification is not None:
            system_prompt = f"[Patch-level Summarization | Magnification={magnification}x]\n" + system_prompt

        user_prompt = (
            f"--- Patch Descriptions (with coordinates) ---\n{chunk_text}\n"
            "--- End of Descriptions ---\n\n"
        )

        if question_text:
            user_prompt += f"Related Question: {question_text}\n\n"

        user_prompt += (
            "Summarize the main pathological findings across these patches. "
            "Your summary should:\n"
            "- Capture key morphological features and cellular details.\n"
            "- If a question is provided, emphasize features that are relevant to answering it.\n"
            "- Do not provide a final answer or diagnosis.\n"
            "Output only the summary text, no JSON or extra formatting."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        with torch.no_grad():
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        summary_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Release memory
        del model_inputs, generated_ids, output_ids
        torch.cuda.empty_cache()

        summaries.append(
            f"[Chunk {i//chunk_size+1} | Patches={coords_str}]\n{summary_text}"
        )

        print(f"Completed summary for chunk {i//chunk_size+1} (patch count: {len(chunk_names)})")

    # Concatenate summaries of all chunks
    combined_summary = (
        f"[Current Magnification: {magnification}x]\n\n" +
        "\n\n".join(summaries)
    )
    return combined_summary
