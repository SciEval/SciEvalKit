import os
import os.path as osp
import re
import json
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from vlmeval import load, dump
from ..smp.file import get_intermediate_file_path, HFCacheRoot
from .text_base import TextBaseDataset


# ------------------ Environment Setup ------------------
# nltk.data.path.append('/mnt/shared-storage-user/sdpdev-fs/sunhaoran/Model/nltk_data')
# print("Loading all-mpnet-base-v2...")
# EMBEDDING_MODEL = SentenceTransformer('/mnt/shared-storage-user/sdpdev-fs/sunhaoran/Model/all-mpnet-base-v2')
# print("Loading all-MiniLM-L6-v2...")
# KEYWORD_MODEL = KeyBERT(SentenceTransformer('/mnt/shared-storage-user/sdpdev-fs/sunhaoran/Model/all-MiniLM-L6-v2'))
# print("Done!")

# 1️⃣ 确定统一缓存根路径
HF_ROOT = HFCacheRoot()
print(f"[INFO] Using HuggingFace cache root: {HF_ROOT}")

# 2️⃣ 设置 NLTK 数据目录（放在相同根路径下）
NLTK_DIR = osp.join(HF_ROOT, "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
print(f"[INFO] NLTK data directory: {NLTK_DIR}")

# 3️⃣ 如果需要下载（只有第一次）
for pkg in ["punkt", "wordnet"]:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == "punkt" else f'corpora/{pkg}')
    except LookupError:
        print(f"[INFO] Downloading missing NLTK resource: {pkg} → {NLTK_DIR}")
        nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)

# 4️⃣ 模型加载路径（放在相同根目录下）
MODEL_DIR = osp.join(HF_ROOT, "sentence_transformers")
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"[INFO] SentenceTransformer model cache: {MODEL_DIR}")

# 5️⃣ 加载模型
print("[INFO] Loading all-mpnet-base-v2 ...")
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2', cache_folder=MODEL_DIR)

print("[INFO] Loading all-MiniLM-L6-v2 ...")
KEYWORD_MODEL = KeyBERT(SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_DIR))

print("[INFO] All models and resources loaded successfully.")

SIMILARITY_THRESHOLD = 0.7
# ------------------ Helper Functions ------------------
def extract_text_response(text: str) -> str:
    """提取 [ANSWER_START]...[ANSWER_END] 之间的内容。"""
    text = text.split('</think>')[-1]
    parts = re.split(r'\[ANSWER_START\]|\[ANSWER_END\]', text)
    if len(parts) >= 3:
        return parts[1].strip()
    # fallback: 删除额外标签
    return text.strip()


def compute_text_generation_metrics(reference: str, generated: str):
    """计算 BLEU、METEOR、ROUGE。"""
    ref_tokens = nltk.word_tokenize(reference.lower())
    gen_tokens = nltk.word_tokenize(generated.lower())

    bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5),
                         smoothing_function=SmoothingFunction().method1)
    meteor = meteor_score([ref_tokens], gen_tokens)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)

    return {
        "bleu": bleu,
        "meteor": meteor,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure
    }


def compute_keyword_overlap(ref_text: str, gen_text: str, top_k=64):
    """计算关键词精确率、召回率、F1。"""
    ref_kw = set([kw for kw, _ in KEYWORD_MODEL.extract_keywords(ref_text, top_n=top_k)])
    gen_kw = set([kw for kw, _ in KEYWORD_MODEL.extract_keywords(gen_text, top_n=top_k)])

    if not ref_kw or not gen_kw:
        return 0.0, 0.0, 0.0

    inter = ref_kw & gen_kw
    p = len(inter) / len(gen_kw)
    r = len(inter) / len(ref_kw)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def compute_step_recall_and_redundancy(reference_steps, generated_steps):
    """计算步骤召回（step recall）和冗余惩罚（redundancy penalty）。"""
    if not reference_steps or not generated_steps:
        return 0.0, 1.0

    ref_embeds = EMBEDDING_MODEL.encode(reference_steps)
    gen_embeds = EMBEDDING_MODEL.encode(generated_steps)

    matched_refs = set()
    matched_gens = set()

    # Step recall
    for i, ref_vec in enumerate(ref_embeds):
        for j, gen_vec in enumerate(gen_embeds):
            if cosine_similarity([ref_vec], [gen_vec])[0][0] >= SIMILARITY_THRESHOLD:
                matched_refs.add(i)
                break

    # Redundancy
    for i, gen_vec in enumerate(gen_embeds):
        for j, ref_vec in enumerate(ref_embeds):
            if cosine_similarity([gen_vec], [ref_vec])[0][0] >= SIMILARITY_THRESHOLD:
                matched_gens.add(i)
                break

    sr = len(matched_refs) / len(reference_steps)
    rp = 1.0 - ((len(generated_steps) - len(matched_gens)) / len(generated_steps))
    return sr, rp


# ------------------ Main Dataset Class ------------------
class BioProBench_GEN(TextBaseDataset):
    """BioProBench-GEN: Protocol Generation Evaluation."""

    MODALITY = "TEXT"
    TYPE = "GEN"
    DATASET_URL = {
        "GEN": "GEN_test.tsv"
    }
    DATASET_MD5 = {
        "GEN": "0f8f1147f8a572592ea47bd91a469276"
    }

    # ---------- Prompt Builder ----------
    def build_prompt(self, line):
        """
        构建 BioProBench-GEN 的 Prompt。
        支持 system_prompt + instruction + input。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        system_prompt = line.get("system_prompt", "")
        instruction = line.get("instruction", "")
        input_text = line.get("input", "")

        prompt = f"""{system_prompt}
{instruction}
Format requirements:
- Each step must be on a separate line.

{input_text}""".strip()

        msgs = [{"type": "text", "value": prompt}]
        return msgs

    # ---------- Evaluation ----------
    def evaluate(self, eval_file: str, **judge_kwargs) -> pd.DataFrame:
        """
        Evaluate protocol generation task predictions.
        输入字段：
            'system_prompt', 'instruction', 'input', 'output', 'prediction'
        输出：
            BLEU, METEOR, ROUGE-1/2/L, KW-F1, Step Recall, Redundancy。
        """
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                data = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported JSON dict structure")

        assert all(k in data for k in ["output", "prediction"]), \
            "eval_file must contain 'output' (ground truth) and 'prediction'"

        score_file = get_intermediate_file_path(eval_file, "_score", "csv")
        storage = get_intermediate_file_path(eval_file, "_judge", "json")


        if osp.exists(storage):
            results = load(storage)
        else:
            bleu_list, meteor_list, rouge1_list, rouge2_list, rougel_list = [], [], [], [], []
            kw_p_list, kw_r_list, kw_f1_list, sr_list, rp_list = [], [], [], [], []
            failed = 0

            for _, item in tqdm(data.iterrows(), total=len(data), desc="Evaluating BioProBench-GEN"):
                ref = item["output"]
                gen = item["prediction"]
                if not gen:
                    failed += 1
                    continue

                try:
                    gen_clean = extract_text_response(str(gen))

                    # Step-level metrics
                    if isinstance(ref, list):
                        ref_steps = [s.strip() for s in ref if s.strip()]
                        gen_steps = [s.strip() for s in gen_clean.split("\n") if s.strip()]
                        sr, rp = compute_step_recall_and_redundancy(ref_steps, gen_steps)
                        sr_list.append(sr)
                        rp_list.append(rp)
                        ref_text = " ".join(ref_steps)
                    else:
                        ref_text = str(ref)
                    gen_text = str(gen_clean)

                    # Text metrics
                    t_metrics = compute_text_generation_metrics(ref_text, gen_text)
                    bleu_list.append(t_metrics["bleu"])
                    meteor_list.append(t_metrics["meteor"])
                    rouge1_list.append(t_metrics["rouge1"])
                    rouge2_list.append(t_metrics["rouge2"])
                    rougel_list.append(t_metrics["rougeL"])

                    # Keyword metrics
                    kw_p, kw_r, kw_f1 = compute_keyword_overlap(ref_text, gen_text)
                    kw_p_list.append(kw_p)
                    kw_r_list.append(kw_r)
                    kw_f1_list.append(kw_f1)

                except Exception:
                    failed += 1

            results = {
                "BLEU": np.mean(bleu_list),
                "METEOR": np.mean(meteor_list),
                "ROUGE-1": np.mean(rouge1_list),
                "ROUGE-2": np.mean(rouge2_list),
                "ROUGE-L": np.mean(rougel_list),
                "KW_Precision": np.mean(kw_p_list),
                "KW_Recall": np.mean(kw_r_list),
                "KW_F1": np.mean(kw_f1_list),
                "Step_Recall": np.mean(sr_list) if sr_list else None,
                "Redundancy_Penalty": np.mean(rp_list) if rp_list else None,
                "Failed_Rate": failed / len(data),
                "Total": len(data),
            }
            dump(results, storage)

        res = pd.DataFrame({
            "Metric": list(results.keys()),
            "Value": list(results.values())
        })
        dump(res, score_file)
        return res
