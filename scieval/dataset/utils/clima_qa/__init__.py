# 导出公共接口
from .base_metric import BaseMetric
from .bleu_score import BLEUScore
from .bert_score import BERTScore
from .fa_score import FAScore

__all__ = [
    "BaseMetric",
    "BLEUScore",
    "BERTScore",
    "FAScore",
]
