# -*- coding: utf-8 -*-
import os
import re
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scieval.smp import *
from ..text_base import TextBaseDataset

class ResearchbenchRank(TextBaseDataset):
    MODALITY = 'TEXT'
    TYPE = 'MCQ'
    NAME = 'ResearchbenchRank'

    PROMPT_TMPL = (
        "You are assisting scientists with their research. Given a research question and two research hypothesis "
        "candidates proposed by large language models, your task is to predict which hypothesis is a better research "
        "hypothesis. By 'better', we mean the hypothesis is more valid and effective for the research question.\n"
        "Please note:\n"
        "(1) Neither hypothesis has been tested experimentally. Ignore any claimed expected performance; only focus on "
        "the technical content and predict which would be more effective if tested in real experiments.\n"
        "(2) We only care about the fundamental core idea. Extra details or complexity are neither advantages nor disadvantages.\n\n"
        "Research question: {bg}\n"
        "Research hypothesis candidate 1: {h1}\n"
        "Research hypothesis candidate 2: {h2}\n\n"
        "Use EXACTLY this format:\n"
        "**Analysis**: <brief reasoning>\n"
        "**Selection of research hypothesis candidate**: candidate 1 or candidate 2"
    )

    DATASET_URL = {
        'ResearchbenchRank': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/ResearchbenchRank.tsv'
    }

    DATASET_MD5 = {
        'ResearchbenchRank': 'fb725d13c7da926de01ffa1802a1c4e7'
    }

    MAIN_PATTERN = re.compile(
        r"\*\*Selection\s+of\s+research\s+hypothesis\s+candidate\*\*\s*[:：]\s*candidate\s*(\d+)",
        flags=re.IGNORECASE,
    )
    FALLBACK_PATTERNS = [
        re.compile(r"candidate\s*([12])", flags=re.IGNORECASE),
        re.compile(r"\b([12])\b"),
    ]

    def __init__(self, dataset: str, ann_path: str | None = None, save_dir: Optional[str] = None, **kwargs):
        self.dataset = dataset
        self.dataset_name = dataset
        self.prepare_tsv(self.DATASET_URL[dataset], self.DATASET_MD5[dataset])
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = self.DATASET_URL[self.dataset_name].split('/')[-1]
        data_path = osp.join(data_root, file_name)
        self.ann_path = ann_path if ann_path is not None else data_path
        self.save_dir = save_dir
        self.dump_image = False
        if not self.ann_path or not os.path.exists(self.ann_path):
            raise FileNotFoundError(f"[ResearchbenchRank] TSV not found: {self.ann_path}")
        raw = pd.read_csv(self.ann_path, sep="\t", dtype=str).fillna("")
        required = [
            "Background Question",
            "Main hypothesis",
            "fake generate hypothesis",
            "model generate hypothesis",
            "DOI",
            "subject",
        ]
        for c in required:
            if c not in raw.columns:
                raise ValueError(f"[ResearchbenchRank] Missing column in TSV: {c}")
        def parse_list_field(x: str) -> List[str]:
            try:
                v = json.loads(x) if isinstance(x, str) and x.strip() else []
                if not isinstance(v, list):
                    return []
                return [str(s).strip() for s in v if isinstance(s, (str, int, float)) and str(s).strip()]
            except Exception:
                return []
        pairs: List[Dict[str, Any]] = []
        for _, r in raw.iterrows():
            bg = str(r["Background Question"]).strip()
            h1 = str(r["Main hypothesis"]).strip()
            doi = str(r["DOI"]).strip()
            subj = str(r["subject"]).strip()
            fakes = parse_list_field(r["fake generate hypothesis"])
            models = parse_list_field(r["model generate hypothesis"])
            cands = fakes + models
            if not bg or not h1 or len(cands) == 0:
                continue
            for k, h2 in enumerate(cands, start=1):
                pairs.append(dict(
                    DOI=doi,
                    subject=subj,
                    bg=bg,
                    hypothesis_1=h1,
                    hypothesis_2=h2,
                    pair_idx=k,
                    question=self.PROMPT_TMPL.format(bg=bg, h1=h1, h2=h2),
                    options='["candidate 1","candidate 2"]',
                    answer="candidate 1",
                ))
        if len(pairs) == 0:
            raise RuntimeError("[ResearchbenchRank] No pair-level samples could be constructed from TSV.")
        self.df = pd.DataFrame(pairs)
        self.df.insert(0, "index", range(1, len(self.df) + 1))
        self.data = self.df

    def __len__(self):
        return len(self.data)
    def build_prompt(self, line: Union[int, pd.Series]) -> List[Dict[str, Any]]:
        row = self.df.iloc[line] if isinstance(line, int) else line
        return [dict(type="text", value=str(row["question"]))]
    def get_answer(self, idx: int) -> str:
        return str(self.df.iloc[idx]["answer"])
    def _parse_choice(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {"choice": None, "why": "prediction_not_str"}
        s = text.strip()
        m = self.MAIN_PATTERN.search(s)
        if m:
            v = m.group(1)
            if v in ("1", "2"):
                return {"choice": int(v), "why": "main_pattern"}
        low = s.lower()
        for pat in self.FALLBACK_PATTERNS:
            m2 = pat.search(low)
            if m2:
                v = m2.group(1)
                if v in ("1", "2"):
                    return {"choice": int(v), "why": f"fallback:{pat.pattern}"}
        if low in ("candidate 1", "candidate1", "1"):
            return {"choice": 1, "why": "strict_fallback_1"}
        if low in ("candidate 2", "candidate2", "2"):
            return {"choice": 2, "why": "strict_fallback_2"}
        return {"choice": None, "why": "unparsed"}
    def evaluate(self, eval_file: str, **judge_kwargs) -> Dict[str, Any]:
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"[ResearchbenchRank] eval_file not found: {eval_file}")
        df_pred, errors = None, []
        for loader in (lambda p: pd.read_excel(p),
                       lambda p: pd.read_csv(p),
                       lambda p: pd.read_csv(p, sep="\t")):
            try:
                tmp = loader(eval_file)
                if "prediction" in tmp.columns and "index" in tmp.columns:
                    df_pred = tmp
                    break
            except Exception as e:
                errors.append(str(e))
        if df_pred is None:
            raise RuntimeError(f"[ResearchbenchRank] cannot load predictions, errors={errors}")
        need_join_cols = ["DOI", "subject", "bg", "hypothesis_1", "hypothesis_2", "pair_idx", "answer"]
        if any(c not in df_pred.columns for c in need_join_cols):
            df_pred = df_pred.merge(self.df[["index"] + need_join_cols], on="index", how="left")
        parsed_choice, hit, log = [], [], []
        for _, r in df_pred.iterrows():
            parsed = self._parse_choice(r.get("prediction", ""))
            choice = parsed["choice"]
            parsed_choice.append(choice)
            log.append(parsed["why"])
            hit.append(1 if choice == 1 else 0 if choice in (1, 2) else 0)
        df_pred["parsed_choice"] = parsed_choice
        df_pred["hit"] = hit
        df_pred["log"] = log
        parsable = df_pred["parsed_choice"].notnull()
        num_parsable = int(parsable.sum())
        num_pairs = len(df_pred)
        pairwise_acc = float(df_pred.loc[parsable, "hit"].mean()) if num_parsable > 0 else 0.0
        agg_rows: List[Dict[str, Any]] = []
        for doi, g in df_pred.groupby("DOI"):
            subject = str(g["subject"].iloc[0]) if "subject" in g.columns else ""
            total = int(len(g))
            wins = int((g["parsed_choice"] == 1).sum())
            losses = int((g["parsed_choice"] == 2).sum())
            rank_position = losses + 1
            rank16 = 16 - losses
            rank_score = 1.0 - rank_position / 16.0
            pair_acc_actual = wins / total if total > 0 else 0.0
            agg_rows.append(dict(
                DOI=doi, subject=subject, total_pairs=total, wins=wins, losses=losses,
                rank_position=rank_position, rank16=rank16, rank_score=rank_score,
                pair_acc_actual=pair_acc_actual,
            ))
        df_agg = pd.DataFrame(agg_rows)
        overall = dict(
            num_pairs=num_pairs,
            num_parsable=num_parsable,
            pairwise_acc=pairwise_acc,
            mean_rank_position=float(df_agg["rank_position"].mean()) if len(df_agg) else float('nan'),
            mean_rank16=float(df_agg["rank16"].mean()) if len(df_agg) else float('nan'),
            mean_rank_score=float(df_agg["rank_score"].mean()) if len(df_agg) else float('nan'),
        )
        by_subject = (
            df_agg.groupby("subject")
            .agg(
                num_items=("DOI", "count"),
                mean_rank_position=("rank_position", "mean"),
                mean_rank16=("rank16", "mean"),
                mean_rank_score=("rank_score", "mean"),
            ).reset_index().to_dict("records")
            if len(df_agg) else []
        )
        export_details = bool(judge_kwargs.get("export_details", True))
        if export_details:
            try:
                out_dir = self.save_dir or os.path.dirname(eval_file)
                os.makedirs(out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(eval_file))[0]
                df_agg.to_csv(os.path.join(out_dir, f"{base}__rank_per_doi.csv"), index=False, encoding="utf-8-sig")
                df_pred[[
                    "index","DOI","subject","pair_idx","bg","hypothesis_1","hypothesis_2",
                    "prediction","parsed_choice","hit","log"
                ]].to_csv(os.path.join(out_dir, f"{base}__rank_per_pair.csv"), index=False, encoding="utf-8-sig")
            except Exception:
                pass
        return {"overall": overall}