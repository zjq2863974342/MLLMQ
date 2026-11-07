import datetime
import json
import os
from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}


replace_prompt = " Please answer yes or no."


def mme_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def parse_pred_ans(pred_ans):
    """Brought from Otter Eval"""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    elif len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def mme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["answer"].lower().strip().replace(".", "")
    assert gt_ans in ["yes", "no"]
    assert pred_ans in ["yes", "no", "other"]
    score = 1.0 if pred_ans == gt_ans else 0.0
    category = doc["category"]
    key_name = "mme_percetion_score" if category in eval_type_dict["Perception"] else "mme_cognition_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {key_name: {"question_id": doc["question_id"], "category": category, "score": score}}


def mme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    category2avg_score = {}
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            assert len(scores) == 2, "MME only supports pairwise evaluation"
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        avg_score = total_score / len(question2scores)
        category2avg_score[category] = avg_score
    for category, avg_score in category2avg_score.items():
        eval_logger.info(f"{category}: {avg_score:.2f}")
    total_score = sum(category2avg_score.values())
    return total_score
import datetime
import json
import os
import pathlib
import re
import statistics

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    # "submission" key removed — textvqa_aggregate_submissions needs args.output_path
    # which lmms-eval simple_evaluate doesn't pass, causing all results.json to be lost.
    return {
        "exact_match": accuracy,
    }


def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_post = ""
    ocr_ref = ""
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        if "ocr" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["ocr"]:
            ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"
    return f"{pre_prompt}{doc['question'].capitalize()}{ocr_ref}{post_prompt}"


def textvqa_aggregate_submissions(results, args):
    """Safe fallback: args.output_path may be None in offline/local eval mode.
    If so, write to tempdir instead of crashing."""
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    try:
        path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    except Exception:
        import tempfile
        path = os.path.join(tempfile.gettempdir(), f"textvqa_submission_{now_date_time}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")


# ────────────────────────────────────────────────────────────────────────────
# SEED-Bench-2 (本地版，seedbench_2_local.yaml)
# 复制自 lmms-eval/tasks/seedbench_2/utils.py，无特殊依赖。
# ────────────────────────────────────────────────────────────────────────────
def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def parse_choice_img(choice, img_token):
    if "jpg" in choice or "png" in choice:
        return img_token
    return choice


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # lmms_eval passes lmms_eval_specific_kwargs (not model_specific_kwargs)
    kwargs = lmms_eval_specific_kwargs or {}
    img_token = kwargs.get("img_token", "<image>")
    post_prompt = kwargs.get("post_prompt", "")
    query = doc["question"]
    query = query.replace("<img>", img_token)
    query += "\n"
    query += f"A. {parse_choice_img(doc['choice_a'], img_token)}\n"
    query += f"B. {parse_choice_img(doc['choice_b'], img_token)}\n"
    query += f"C. {parse_choice_img(doc['choice_c'], img_token)}\n"
    query += f"D. {parse_choice_img(doc['choice_d'], img_token)}"
    if doc["data_type"] == "Image Generation":
        num_img_in_question = len(doc["data_id"]) - 4
        prepend_tokens = [img_token] * num_img_in_question
        query = " ".join(prepend_tokens) + "\n" + query
    return f"{query}\n{post_prompt}"


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"].split(" ")
    data_type = "_".join(data_type)

    return {f"seed_{data_type}": {"pred": pred, "answer": answer, "question_id": doc["question_id"]},
            "seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}}


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"] == result["answer"]:
            total_correct += 1
        total_count += 1
    return total_correct / total_count if total_count != 0 else 0


def seed_aggregation_result_all(results):
    score = seed_aggregation_result(results)
    stored_results = []
    for result in results:
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")
    return score


# ── POPE (hallucination detection) ──
def pope_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def pope_doc_to_text(doc):
    question = doc["question"].strip()
    return f"{question}\nAnswer the question using a single word or phrase."

def pope_process_results(doc, results):
    # Janus 输出 "Yes."/"No."（带句点），InternVL2 输出 "yes"/"no"。上游 lmms-eval 的
    # 精确匹配会把带句点回答全部判错（pope_accuracy≈0）。统一 strip 首尾空格与句点。
    pred = results[0].lower().strip(" .").strip()
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "pope_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }

def pope_aggregate_accuracy(results):
    return sum(r["score"] for r in results) / len(results)

def pope_aggregate_precision(results):
    tp = sum(1 for r in results if r["ground_truth"] == "yes" and r["prediction"] == "yes")
    fp = sum(1 for r in results if r["ground_truth"] == "no" and r["prediction"] == "yes")
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def pope_aggregate_recall(results):
    tp = sum(1 for r in results if r["ground_truth"] == "yes" and r["prediction"] == "yes")
    fn = sum(1 for r in results if r["ground_truth"] == "yes" and r["prediction"] == "no")
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def pope_aggregate_f1_score(results, args=None):
    p = pope_aggregate_precision(results)
    r = pope_aggregate_recall(results)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def pope_aggregate_yes_ratio(results):
    yes_count = sum(1 for r in results if r["prediction"] == "yes")
    return yes_count / len(results)


# ── MathVista (math reasoning, works with quick_extract locally) ──
def _get_mathvista_evaluator():
    import sys, os
    _mbq_lmms = os.path.join(os.path.dirname(__file__), "..", "..", "MBQ", "3rdparty", "lmms-eval")
    if _mbq_lmms not in sys.path:
        sys.path.insert(0, _mbq_lmms)
    from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator
    from pathlib import Path
    import yaml
    import json
    cfg_yaml = Path(_mbq_lmms) / "lmms_eval/tasks/mathvista/mathvista.yaml"
    with open(cfg_yaml) as f:
        raw = f.readlines(); safe = [l for l in raw if "!function" not in l]
    cfg = yaml.safe_load("".join(safe))
    api_key = os.getenv("OPENAI_API_KEY", "")
    # Always use quick_extract=True — local regex extraction works for integer/float/
    # multi_choice/text. OpenAI API is only a fallback for unruly formats.
    # This avoids hanging on unreachable OpenAI (502 / timeout) in offline envs.
    return MathVistaEvaluator(
        api_key=api_key,
        gpt_model=cfg["metadata"]["gpt_eval_model_name"],
        quick_extract=True)

def mathvista_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]

def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {"shot": 0, "shot_type": "format-prompt", "use_caption": False, "use_ocr": False}
    ev = _get_mathvista_evaluator()
    # Merge doc top-level keys + metadata sub-dict for create_one_query
    md = doc.get("metadata", {}) or {}
    caption = doc.get("caption", "") or md.get("caption", "")
    ocr = doc.get("ocr", "") or md.get("ocr", "")
    problem = {
        "question_type": doc["question_type"], "answer_type": doc["answer_type"],
        "question": doc["question"], "unit": doc.get("unit") or "",
        "caption": caption, "ocr": ocr,
        "choices": doc.get("choices"), "answer": doc.get("answer"),
        "precision": doc.get("precision", 0),
    }
    return ev.create_one_query(problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"])

def mathvista_process_results(doc, results):
    ev = _get_mathvista_evaluator()
    # Janus 生成数值型答案时常带末尾句点（"0."/"100."/"28.3."），导致 extract_answer
    # 里 int()/float() 转换失败、quick_extract 末尾数字正则 `(\d+)\s*$` 匹配不到 → 提取为空。
    # 这里先 strip 末尾句点，与 Pope 的修复同理。
    pred = results[0].strip().rstrip('.').strip()
    # Use the evaluator's own quick_extract flag (not the yaml config),
    # because we always pass quick_extract=True from _get_mathvista_evaluator()
    quick = getattr(ev, 'quick_extract', True)
    problem = {
        "question_type": doc["question_type"], "answer_type": doc["answer_type"],
        "query": doc.get("query", ""), "choices": doc.get("choices"),
        "answer": doc.get("answer"), "precision": doc.get("precision", 0),
    }
    extraction = ev.extract_answer(pred, problem, quick)
    prediction = ev.normalize_extracted_answer(extraction, problem.get("choices"),
        problem["question_type"], problem["answer_type"], problem["precision"])
    true_false = ev.safe_equal(prediction, problem["answer"]) if problem.get("answer") is not None else False
    return {
        "mathvista_acc": {
            "question_id": doc["pid"], "query": doc.get("query", doc.get("question", "")),
            "choices": doc.get("choices"), "answer": doc.get("answer"),
            "extraction": extraction, "prediction": prediction, "true_false": true_false,
            "question_type": doc["question_type"], "answer_type": doc["answer_type"],
            "precision": doc.get("precision", 0), "metadata": doc["metadata"],
        }
    }

def mathvista_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    total = len(results)
    correct = sum(1 for r in results if r.get("true_false"))
    return round(correct / total * 100, 2) if total > 0 else 0.0
