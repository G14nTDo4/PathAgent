import os
import json
import hashlib
import difflib
import argparse
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

# ====== Utility Functions ======
def make_unique_id(long_id, question_text):
    q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
    return f"{long_id}_{q_hash}"

def compute_scores(gts, res):
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]
    eval_res = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def acc_of_seq(choices, gt, res):
    """Determine if prediction 'res' is most similar to 'gt' among choices."""
    if not choices:
        return None
    score = difflib.SequenceMatcher(None, res, gt).quick_ratio()
    for item in choices:
        tmp = difflib.SequenceMatcher(None, res, item).quick_ratio()
        if tmp > score:
            return False
    return True

# ====== Main Evaluation Process ======
def evaluate_results(results_dir):
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return None, None, None, None

    gts = {}
    res = {}
    
    total_mcq = 0
    correct_mcq = 0

    file_list = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

    for fname in file_list:
        fpath = os.path.join(results_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        case_id = make_unique_id(data["long_id"], data["question"])
        gt = data.get("ground_truth", "").strip()
        pred = data.get("pred_answer", "").strip()
        choices = data.get("choices", None)

        if gt and pred:
            gts[case_id] = [gt]
            res[case_id] = [pred]

        if choices and isinstance(choices, list) and len(choices) > 0:
            total_mcq += 1
            if acc_of_seq(choices, gt, pred):
                correct_mcq += 1

    print(f"Collected {len(gts)} valid samples")

    # Compute BLEU / ROUGE / METEOR
    nlu_scores = compute_scores(gts, res)

    # Compute MCQ Accuracy
    acc_score = correct_mcq / total_mcq if total_mcq > 0 else 0.0

    print(f"Total MCQ: {total_mcq}, Correct: {correct_mcq}, Accuracy = {acc_score:.4f}")

    return nlu_scores, acc_score, gts, res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Results")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Path to the directory containing result JSON files")
    
    args = parser.parse_args()

    print(f"Evaluating results in: {args.results_dir}")
    
    scores, acc, gts, res = evaluate_results(args.results_dir)

    if scores:
        print("\nNLU Evaluation Results:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
        print(f"\nMCQ Accuracy: {acc:.4f}")