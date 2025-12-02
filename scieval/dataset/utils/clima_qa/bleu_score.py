from typing import List
from .base_metric import BaseMetric
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# 首次使用需要：
# import nltk
# nltk.download('punkt')

_smooth = SmoothingFunction().method1

class BLEUScore(BaseMetric):
    """
    与你给的论文代码一致的实现思路：
    - 句级：把 (ref, hyp) 组成长度为1的语料，调用 corpus_bleu
    - 语料级：refs 为 [[ref1], [ref2], ...]，hyps 为 [hyp1, hyp2, ...]
    """

    def get_sentence_score(self, reference_answer: str, generated_answer: str) -> float:
        references = [[reference_answer]]
        candidates = [generated_answer]
        return float(corpus_bleu(references, candidates, smoothing_function=_smooth))

    def get_corpus_score(self, reference_answer_corpus: List[str], generated_answer_corpus: List[str]) -> float:
        if not reference_answer_corpus:
            return 0.0
        references = [[ref] for ref in reference_answer_corpus]
        return float(corpus_bleu(references, generated_answer_corpus, smoothing_function=_smooth))
