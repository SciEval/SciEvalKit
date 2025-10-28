"""
FA Score（Fact Alignment / Faithfulness-like）：
- 与你提供的代码保持一致的思路：用 OpenAI Chat Completions 计算 SUPPORTS vs REFUTES 的对数概率比，
  再做 sigmoid 映射得到 [0,1] 分数；可选温度平滑。
- 工程增强：
  1) 可通过 use_openai=False 关闭（返回 None 或 0）
  2) 容错：无 API Key 或网络异常时不崩溃
"""

from typing import List, Optional, Tuple
from .base_metric import BaseMetric
import numpy as np
from ..judge_util import build_judge
import os

try:
    from openai import OpenAI
    import tiktoken
except Exception:
    OpenAI = None
    tiktoken = None


LABELS = ["SUPPORTS", "REFUTES"]

_LOGIT_SYS_PROMPT = f"""
You are a climate expert that annotates if a given claim either SUPPORTS or REFUTES the presented evidence.
You will be provided the following as the input:

Evidence: <evidence>
Claim: <claim>

Respond with just one word - {LABELS[0]} if the claim supports the evidence and {LABELS[1]} otherwise.
""".strip()

_USER_PROMPT = "Evidence: {}\nClaim: {}"

def _smooth_probability(p: float, T: float = 5.0) -> float:
    if p <= 0.0 or p >= 1.0:
        return float(p)
    logit = np.log(p / (1.0 - p))
    smooth_logit = logit / T
    return float(1.0 / (1.0 + np.exp(-smooth_logit)))


class FAScore(BaseMetric):
    """
    Args:
        openai_model: OpenAI 模型名，如 'gpt-4o-mini','gpt-4' 或微调名。
        use_openai: 为 False 时，get_* 返回 0.0（或 None），用于离线跑通流程。
        return_none_when_disabled: True 时禁用则返回 None；否则返回 0.0。
        temperature_smooth_T: 平滑温度 T，默认 5 与论文代码一致。
    """
    def __init__(self,
                 openai_model: str = "gpt-4",
                 use_openai: bool = True,
                 return_none_when_disabled: bool = False,
                 temperature_smooth_T: float = 5.0):
        self.use_openai = use_openai and (OpenAI is not None) and (tiktoken is not None)
        self.return_none_when_disabled = return_none_when_disabled
        self.temperature_smooth_T = float(temperature_smooth_T)

        self.model_name = openai_model # TODO need to be fixed
        self.model = build_judge(model=self.model_name) # TODO need to be fixed
        if self.use_openai:
            try:
                base_url = self.model.api_base[:-17]
                self.client = OpenAI(base_url=base_url) # TODO need to be fixed
                encoding_model = openai_model.split(':')[1] if openai_model[:2] == 'ft' else openai_model
                self.encoder = tiktoken.encoding_for_model(encoding_model)
            except Exception:
                # 回退为禁用模式
                self.use_openai = False
                self.client = None
                self.encoder = None
        else:
            self.client = None
            self.encoder = None

    def _disabled_return(self):
        return None if self.return_none_when_disabled else 0.0

    def _score_one(self, evidence: str, claim: str) -> float:
        """
        与原代码一致：取 SUPPORTS 与 REFUTES 的 top_logprobs，做比值 -> 概率 -> 平滑。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                logprobs=True,
                top_logprobs=20,
                messages=[
                    {"role": "system", "content": _LOGIT_SYS_PROMPT},
                    {"role": "user", "content": _USER_PROMPT.format(evidence, claim)},
                ],
            )
            result = response.choices[0].logprobs.content[0].top_logprobs

            label_tokens = [self.encoder.decode(self.encoder.encode(label)[:1]) for label in LABELS]
            label_logprobs = [-100.0, -100.0]
            for i in range(len(label_logprobs)):
                for res in result:
                    if res.token == label_tokens[i]:
                        label_logprobs[i] = float(res.logprob)

            ratio = np.exp(label_logprobs[0] - label_logprobs[1])  # SUPPORTS vs REFUTES
            score = float(ratio / (ratio + 1.0))
            return _smooth_probability(score, T=self.temperature_smooth_T)
        except Exception as e:
            print(e)
            # 网络/额度/权限异常时不给挂——返回 0
            return 0.0

    # ===== BaseMetric 接口 =====

    def get_sentence_score(self, reference_answer: str, generated_answer: str) -> float:
        if not self.use_openai:
            return self._disabled_return()
        # 注意：论文脚本中 Evidence=reference_answer, Claim=generated_answer
        return self._score_one(evidence=reference_answer, claim=generated_answer)

    def get_corpus_score(self, reference_answer_corpus: List[str], generated_answer_corpus: List[str]) -> float:
        if not reference_answer_corpus:
            return 0.0 if not self.return_none_when_disabled else None
        if not self.use_openai:
            return self._disabled_return()

        scores = []
        for ref, hyp in zip(reference_answer_corpus, generated_answer_corpus):
            scores.append(self._score_one(evidence=ref, claim=hyp))
        return float(np.mean(scores))
