from typing import List, Tuple, Any

class BaseMetric:
    """
    统一的指标基类。
    子类至少实现：
      - get_sentence_score(ref: str, hyp: str) -> Any
      - get_corpus_score(refs: List[str], hyps: List[str]) -> Any
    """
    def get_sentence_score(self, reference_answer: str, generated_answer: str) -> Any:
        raise NotImplementedError

    def get_corpus_score(self, reference_answer_corpus: List[str], generated_answer_corpus: List[str]) -> Any:
        raise NotImplementedError
