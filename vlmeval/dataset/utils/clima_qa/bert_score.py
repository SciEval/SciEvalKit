from typing import List, Tuple
from .base_metric import BaseMetric

# bert-score 依赖较重，避免控制台噪音
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

try:
    from bert_score import score as bertscore_compute
except Exception as e:
    bertscore_compute = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


class BERTScore(BaseMetric):
    """
    返回与论文代码一致的指标：
    - 句级：返回 (P, R, F1) 的三元组；evaluate 时通常记录 F1
    - 语料级：返回 F1 的均值
    语言默认 'en'；如需其它语言，可在初始化时传入 lang。
    """
    def __init__(self, lang: str = "en"):
        if bertscore_compute is None:
            raise ImportError(
                f"bert-score 未安装或导入失败：{_IMPORT_ERR}. 请先 `pip install bert-score`"
            )
        self.lang = lang

    def get_sentence_score(self, reference_answer: str, generated_answer: str) -> Tuple[float, float, float]:
        references = [reference_answer]
        candidates = [generated_answer]
        P, R, F1 = bertscore_compute(candidates, references, lang=self.lang)
        return P[0].item(), R[0].item(), F1[0].item()

    def get_corpus_score(self, reference_answer_corpus: List[str], generated_answer_corpus: List[str]) -> float:
        if not reference_answer_corpus:
            return 0.0
        P, R, F1 = bertscore_compute(generated_answer_corpus, reference_answer_corpus, lang=self.lang)
        return float(F1.mean().item())
