# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import os
import pandas as pd
import csv, sys
from ..text_base import TextBaseDataset
from scieval.smp import *

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

class ResearchbenchRetrieve(TextBaseDataset):
    MODALITY = "TEXT"
    TYPE = "TEXT"
    NAME = "ResearchbenchRetrieve"
    DATASET_URL = {
        'ResearchbenchRetrieve': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/ResearchbenchRetrieve.tsv'
    }

    DATASET_MD5 = {
        'ResearchbenchRetrieve': 'ebf4ed3a14b8e0975691d7980b9a93db'
    }
    dataset = NAME
    @property
    def dataset_name(self) -> str:
        return getattr(self, "dataset", self.NAME)
    def __len__(self) -> int:
        return len(self.data)
    def get_prompt(self, idx: int):
        return self.build_prompt(idx)

    def __init__(self,
                 tsv_path: str |None = None,
                 require_eval_flag: bool = True,
                 use_abstract: bool = True,
                 abstract_maxlen: int = 220, dataset: Optional[str] = None):
        self.prepare_tsv(self.DATASET_URL[dataset], self.DATASET_MD5[dataset])
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = self.DATASET_URL[self.dataset_name].split('/')[-1]
        data_path = osp.join(data_root, file_name)
        self.tsv_path = tsv_path if tsv_path is not None else data_path
        self.require_eval_flag = require_eval_flag
        self.use_abstract = use_abstract
        self.abstract_maxlen = abstract_maxlen
        self.dataset = dataset or self.NAME
        self.df = pd.read_csv(self.tsv_path, sep="\t", dtype=str, engine="c")
        needed = [
            "sample_id", "category", "doi", "normalized_doi",
            "use_in_eval", "weight",
            "background_question", "background_survey",
            "candidates_json", "groundtruth_titles"
        ]
        for k in needed:
            if k not in self.df.columns:
                raise ValueError(f"TSV: {k}")
        self.df["use_in_eval"] = self.df["use_in_eval"].fillna("0").astype(str)
        self.df["weight"] = self.df["weight"].fillna("1").astype(int)
        self.df["background_question"] = self.df["background_question"].fillna("")
        self.df["background_survey"] = self.df["background_survey"].fillna("")
        def _loads_safe(x):
            try:
                return json.loads(x)
            except Exception:
                return []
        self.df["candidates"] = self.df["candidates_json"].map(_loads_safe)
        self.df["gt_titles"] = self.df["groundtruth_titles"].map(_loads_safe)

        if self.require_eval_flag:
            mask = self.df["use_in_eval"].str.lower().isin(["1", "true", "yes"])
            self.df_eval = self.df[mask].reset_index(drop=True)
        else:
            self.df_eval = self.df.reset_index(drop=True)

        self.df_eval = self.df_eval.reset_index(drop=True)
        self.df_eval["index"] = self.df_eval.index
        self.data = self.df_eval
    def dump_image(self, idx: int):
        return None
    @property
    def img_root(self):
        return None
    def dump_result(self, results, out_path: str):
        import pandas as pd
        import os
        df = pd.DataFrame(results)
        root, ext = os.path.splitext(out_path)
        csv_path = root + ".csv"
        try:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[dump_result] write CSV failed: {csv_path} -> {e}")
        wrote_main = False
        if ext.lower() == ".xlsx":
            try:
                df.to_excel(out_path, index=False)
                wrote_main = True
            except Exception as e:
                print(f"[dump_result] write XLSX failed: {out_path} -> {e}; fallback to CSV: {csv_path}")
        elif ext.lower() == ".csv":
            try:
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                wrote_main = True
            except Exception as e:
                print(f"[dump_result] write CSV failed: {out_path} -> {e}")
        exist_flags = []
        for p in [out_path, csv_path]:
            exist_flags.append(f"{p}={'OK' if os.path.exists(p) else 'MISS'}")
        print("[dump_result] outputs -> " + " ; ".join(exist_flags))
    @staticmethod
    def _first_round_prompts() -> List[str]:
        return [
            ("You are helping with the scientific hypotheses generation process. We in general split the period of "
             "research hypothesis proposal into three steps. Firstly it's about finding a good and specific background "
             "research question, and an introduction of the previous methods under the same topic; Secondly its about "
             "finding inspirations (mostly from literatures), which combined with the background research question, "
             "can lead to an impactful research hypothesis; Finally it's hypothesis generation based on the background "
             "research question and found inspirations. Usually a paper can be choosed as an inspiration is because it "
             "can potentially help to solve or alleviate one problem of a previous method for this research question "
             "so that leveraging the concepts related to the inspiration, a better method can be developed based on the "
             "previous methods and this inspiration. Take backpropagation as an example, the research question is how to "
             "use data to automatically improve the parameters of a multi-layer logistic regression with data, the "
             "inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. "
             "Here the previous method can only inference the multi-layer logistic regression, but can't automatically "
             "update its parameters to learn from data. The selected chain rule inspiration can be leveraged to "
             "automatically update the parameters in the multi-layer logistic regression, and therefore improve over the "
             "previous method to create hypothesis. \nGiven a research question, the background and some of the existing "
             "methods for this research question, and several top-tier publications (including their title and abstract), "
             "try to identify which publication can potentially serve as an inspiration for the background research "
             "question so that combining the research question and the inspiration in some way, a novel, valid, and "
             "significant research hypothesis can be formed. Now try to select inspirations based on the background "
             "research question. \nThe background research question is: "),
            "\n\nThe introduction of the previous methods is:",
            "\n\nThe potential inspiration candidates are: ",
            ("\n\nNow you have seen the background research question, existing methods, and many potential inspiration "
             "candidates. Please try to identify which three literature candidates are the most possible to serve as the "
             "inspiration to the background research question? Please name the title of the literature candidate, and "
             "also try to give your reasons. (response format: 'Title: \nReason: \nTitle: \nReason: \nTitle: \nReason: \n')")
        ]
    def build_prompt(self, line: int | pd.Series) -> List[Dict[str, str]]:
        if isinstance(line, int):
            row = self.df_eval.iloc[line]
        else:
            row = line
        rq = row["background_question"].strip()
        bs = row["background_survey"].strip()
        cands = row["candidates"]  # [[title, abstract], ...]
        intro_lines = []
        for i, ta in enumerate(cands):
            title = (ta[0] if len(ta) > 0 else "").strip()
            abstract = (ta[1] if len(ta) > 1 else "").strip()
            if not self.use_abstract:
                abstract = ""
            else:
                if self.abstract_maxlen and len(abstract) > self.abstract_maxlen:
                    abstract = abstract[: self.abstract_maxlen].rstrip() + "..."
            seg = (f"Next we will introduce inspiration candidate {i}. "
                   f"Title: {title}; Abstract: {abstract}. "
                   f"The introduction of inspiration candidate {i} has come to an end.\n")
            intro_lines.append(seg)
        cand_block = "".join(intro_lines)
        p = self._first_round_prompts()
        prompt = p[0] + rq + p[1] + bs + p[2] + cand_block + p[3]
        return [dict(type="text", value=prompt)]
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        A, B = set(a.split()), set(b.split())
        if not A or not B:
            return 0.0
        return len(A & B) / max(1, len(A | B))
    @classmethod
    def _best_match(cls, title: str, pool: List[str]) -> Tuple[int, float, str]:
        t = cls._norm(title)
        best_i, best_sim, best_title = -1, -1.0, ""
        for i, p in enumerate(pool):
            s = cls._norm(p)
            v = cls._jaccard(t, s)
            if v > best_sim:
                best_i, best_sim, best_title = i, v, p
        if best_sim < 0.25:
            return -1, best_sim, ""
        return best_i, best_sim, best_title
    @staticmethod
    def _extract_title_reason_blocks(text: str, k: int = 3) -> List[Tuple[str, str]]:
        if not isinstance(text, str):
            return []
        text = re.sub(r"[#*]", "", text).strip()
        if not text.strip().startswith("Title:"):
            m = re.search(r"(^|\n)\s*Title\s*:", text, flags=re.IGNORECASE)
            if m:
                text = text[m.start():]
        blocks = re.split(r"\bTitle\s*:\s*", text, flags=re.IGNORECASE)
        pairs: List[Tuple[str, str]] = []
        for blk in blocks:
            blk = blk.strip()
            if not blk:
                continue
            parts = re.split(r"\bReason\s*:\s*", blk, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) != 2:
                title_line = parts[0].strip()
                title = title_line.splitlines()[0].strip().strip(";").strip()
                reason = ""
            else:
                title = parts[0].strip().strip(";").strip()
                reason = parts[1].strip()
            if title:
                pairs.append((title, reason))
            if len(pairs) >= k:
                break
        return pairs[:k]
    @staticmethod
    def _hit_at_k(pred_title_norms: List[str], gt_titles_norms: List[str], k: int) -> int:
        P = [t for t in pred_title_norms[:k]]
        G = set(gt_titles_norms)
        return 1 if any(p in G for p in P) else 0

    def _fallback_pick_by_scan(self, text: str, cand_titles: List[str], topk: int = 3) -> List[int]:
        text_raw = (text or "").lower()
        text_norm = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", text_raw).strip()
        text_tokens = set(text_norm.split())
        pos_hits = []
        sim_hits = []
        for i, t in enumerate(cand_titles):
            t_norm = re.sub(r"\s+", " ", (t or "").lower()).strip()
            if not t_norm:
                continue
            t_norm2 = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", t_norm).strip()
            if t_norm2 and t_norm2 in text_norm:
                pos = text_norm.find(t_norm2)
                pos_hits.append((pos, i))
                continue
            cand_tokens = set(t_norm2.split())
            if cand_tokens:
                overlap = len(cand_tokens & text_tokens) / max(1, len(cand_tokens))
                if overlap >= 0.5:
                    sim_hits.append((overlap, i))

        picked = []
        pos_hits.sort(key=lambda x: x[0])
        for _, i in pos_hits:
            if i not in picked:
                picked.append(i)
            if len(picked) >= topk:
                return picked
        sim_hits.sort(key=lambda x: x[0], reverse=True)
        for _, i in sim_hits:
            if i not in picked:
                picked.append(i)
            if len(picked) >= topk:
                break
        return picked[:topk]


    def evaluate(
        self,
        eval_file: str,
        save_detail: Optional[str] = None,
        weighted: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        import glob
        path = eval_file
        if not os.path.exists(path):
            root, _ = os.path.splitext(eval_file)
            candidates = [root + ".xlsx", root + ".csv"]
            patt = os.path.join(os.path.dirname(eval_file),
                                os.path.basename(root) + ".*")
            found = sorted(glob.glob(patt))
            for p in candidates + found:
                if os.path.exists(p):
                    path = p
                    break
            else:
                raise FileNotFoundError(
                    f"Prediction file not found. Tried: {eval_file}, "
                    f"{root+'.xlsx'}, {root+'.csv'} and glob: {patt}"
                )
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xlsx":
            pred = pd.read_excel(path)
        elif ext == ".csv":
            pred = pd.read_csv(path)
        elif ext in [".json", ".jsonl"]:
            pred = pd.read_json(path, lines=(ext == ".jsonl"))
        else:
            pred = pd.read_csv(path)
        key = "sample_id" if "sample_id" in pred.columns and "sample_id" in self.df_eval.columns else "index"
        df_eval = self.df_eval.copy().reset_index(drop=False).rename(columns={"index": "_row_id"})
        if key == "index":
            if "index" not in pred.columns:
                raise ValueError("No 'index' column in the predictions file.")
            pred["_row_id"] = pred["index"]
            merged = pd.merge(df_eval, pred, on="_row_id", how="left", suffixes=("", "_pred"))
        else:
            merged = pd.merge(df_eval, pred, on="sample_id", how="left", suffixes=("", "_pred"))
        if "prediction" not in merged.columns:
            raise ValueError("No 'prediction' column in the predictions file.")
        logs, pred_titles_col, hit1_col, hit3_col, w_col = [], [], [], [], []
        for _, r in merged.iterrows():
            cands = r["candidates"]
            cand_titles = [str(x[0]).strip() for x in cands]
            cand_titles_norm = [self._norm(t) for t in cand_titles]
            gts = [str(t).strip() for t in (r["gt_titles"] or [])]
            gt_norms = [self._norm(t) for t in gts]
            text = str(r.get("prediction", ""))
            pairs = self._extract_title_reason_blocks(text, k=3)  # list[(title, reason)]
            chosen_idxs, dbg = [], []
            if pairs:
                for title, _rs in pairs:
                    t_norm = self._norm(title)
                    if t_norm in cand_titles_norm:
                        idx = cand_titles_norm.index(t_norm)
                        if idx not in chosen_idxs:
                            chosen_idxs.append(idx)
                            dbg.append(f"exact:{title}=>#{idx}")
                            continue
                    idx, sim, matched = self._best_match(title, cand_titles)
                    if idx >= 0 and idx not in chosen_idxs:
                        chosen_idxs.append(idx)
                        dbg.append(f"fuzzy:{title}=>#{idx}({matched})@{sim:.2f}")
                    else:
                        dbg.append(f"unmatched:{title}")
            else:
                chosen_idxs = self._fallback_pick_by_scan(text, cand_titles, topk=3)
                dbg.append(f"fallback_scan:{chosen_idxs}")
            pred_titles = [cand_titles[i] for i in chosen_idxs if 0 <= i < len(cand_titles)]
            pred_titles_norm = [self._norm(t) for t in pred_titles]
            pred_titles_col.append(" | ".join(pred_titles))
            h1 = self._hit_at_k(pred_titles_norm, gt_norms, k=1)
            h3 = self._hit_at_k(pred_titles_norm, gt_norms, k=3)
            hit1_col.append(h1)
            hit3_col.append(h3)
            w_col.append(int(r["weight"]))
            logs.append(f"pairs={[(p[0],) for p in pairs]}; chosen={chosen_idxs}; gt={gts}; {', '.join(dbg)}")
        merged["pred_titles"] = pred_titles_col
        merged["hit"] = hit1_col
        merged["hit@1"] = hit1_col
        merged["hit@3"] = hit3_col
        merged["log"] = logs
        if weighted:
            w = pd.Series(w_col, dtype=float)
            hit1 = float((w * merged["hit@1"]).sum() / max(1.0, w.sum()))
            hit3 = float((w * merged["hit@3"]).sum() / max(1.0, w.sum()))
        else:
            hit1 = float(pd.Series(hit1_col).mean())
            hit3 = float(pd.Series(hit3_col).mean())
        out_dir = os.path.dirname(eval_file)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(eval_file))[0]
        save_detail = os.path.join(out_dir, f"{base}_judge.csv")
        if save_detail:
            cols = [
                "sample_id", "category", "doi", "use_in_eval", "weight",
                "pred_titles", "hit@1", "hit@3", "log"
            ]
            out = merged[cols]
            if save_detail.lower().endswith(".xlsx"):
                out.to_excel(save_detail, index=False)
            else:
                out.to_csv(save_detail, index=False, encoding="utf-8-sig")
        result_file_path = os.path.join(out_dir, f"{base}_eval.json")
        eval_result = {
            "dataset": self.NAME,
            "size": len(self.df_eval),
            "weighted": weighted,
            "hit@1": round(hit1, 6),
            "hit@3": round(hit3, 6),
        }
        # 将字典数据写入 JSON 文件
        with open(result_file_path, "w") as f:
            json.dump(eval_result, f, indent=4)
        return eval_result
