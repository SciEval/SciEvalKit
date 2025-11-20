import warnings
import pandas as pd
import re
from abc import abstractmethod
from collections import Counter
from ..smp import *
from .text_base import TextBaseDataset

# 尝试导入 SMACT 库（可选）
try:
    from smact.screening import smact_validity
    SMACT_AVAILABLE = True
except ImportError:
    SMACT_AVAILABLE = False
    warnings.warn("SMACT library not available. SMACT validity checks will be skipped.")


def extract_elements_from_prompt(prompt: str) -> list:
    """
    Extract element symbols from diverse prompt instructions.
    Supported patterns include:
    - composed of
    - that has
    - characterized by
    - with the composition
    - based on
    - featuring
    - whose makeup is
    """
    if pd.isna(prompt) or not prompt:
        return []
    
    prompt = str(prompt).strip()
    
    patterns = [
        r'composed of',
        r'that has',
        r'characterized by',
        r'with the composition',
        r'based on',
        r'featuring',
        r'whose makeup is'
    ]
    
    joined = '|'.join(patterns)
    match = re.search(rf'(?:{joined})\s+(.*?)(?:[\.。\n]|$)', prompt, re.IGNORECASE)
    
    if match:
        elements_str = match.group(1)
        elements = [
            el.strip() for el in re.split(r'[,\s]+', elements_str)
            if re.fullmatch(r'[A-Z][a-z]?', el.strip())
        ]
        return elements
    
    # fallback: 尝试提取所有可能的元素符号
    fallback = re.findall(r'\b[A-Z][a-z]?\b', prompt)
    # 过滤掉常见的非元素词
    non_elements = {'The', 'A', 'An', 'For', 'With', 'That', 'This', 'Is', 'Are', 'In', 'On', 'At'}
    fallback = [e for e in fallback if e not in non_elements]
    return fallback


def material_postprocessor(text: str) -> str:
    """提取 <material> 标签内容"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    match = re.search(r"<material>(.*?)</material>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def composition_precision(elements: list, prediction: str) -> float:
    """计算元素命中率"""
    if not elements:
        return 0.0
    
    E_pi = set(elements)
    clean = re.sub(r'<[^>]+>', ' ', str(prediction))
    E_gi = set(re.findall(r'\b[A-Z][a-z]?\b', clean))
    
    if not E_pi:
        return 0.0
    
    return len(E_pi & E_gi) / len(E_pi)


class Composition2Material(TextBaseDataset):
    TYPE = 'TEXT'
    DATASET_URL = {
        'composition2material': '/root/code/VLMEvalKit/LMUData/composition2material_no_coords_train.tsv',
    }
    DATASET_MD5 = {
        'composition2material': ''
    }  # MD5码暂时不需要，先置空
    material_generation_task = ['composition2material']
    
    def __init__(self, dataset='composition2material', **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        # 加载 ground truth materials 用于新颖性检查
        self.gt_materials = self._load_gt_materials()
    
    def _load_gt_materials(self):
        """加载训练数据中的所有材料组成，用于新颖性检查"""
        gt_materials = set()
        if 'answer' in self.data.columns:
            for _, row in self.data.iterrows():
                answer = row.get('answer', '')
                mat = material_postprocessor(answer)
                if mat:
                    gt_materials.add(mat.strip())
        return gt_materials
    
    def load_data(self, dataset):
        """
        重写 load_data 方法，直接加载本地文件，而不通过 prepare_tsv 下载
        """
        url = self.DATASET_URL[dataset]
        # 如果是本地文件路径且存在，直接加载
        if osp.exists(url) and osp.isfile(url):
            from ..smp.file import load
            return load(url)
        else:
            # 否则使用父类的方法（尝试下载）
            file_md5 = self.DATASET_MD5[dataset]
            return self.prepare_tsv(url, file_md5)
    
    @staticmethod
    def extract_material_composition(text):
        """
        从文本中提取材料组成。
        格式: <material>元素1 元素2 ... <sg> <sg编号></material>
        """
        if pd.isna(text) or text is None:
            return []
        
        text = str(text).strip()
        
        # 尝试提取 <material>...</material> 标签内的内容
        material_pattern = r'<material>(.*?)</material>'
        match = re.search(material_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1)
            # 移除所有 <sg> 相关标签（格式：<sg> <sg1> 或 <sg> <sg12> 等）
            content = re.sub(r'<sg>\s*<sg\d+>', '', content)
            content = re.sub(r'<sg\d+>', '', content)  # 额外处理单独的 sg 标签
            content = re.sub(r'<sg>', '', content)  # 移除单独的 <sg> 标签
            # 分割元素（元素用空格分隔）
            elements = [e.strip() for e in content.split() if e.strip() and not e.strip().startswith('<')]
            return elements
        else:
            # 如果没有 <material> 标签，尝试提取化学元素（两个大写字母或一个大写字母+小写字母）
            element_pattern = r'\b([A-Z][a-z]?)\b'
            elements = re.findall(element_pattern, text)
            # 过滤掉常见的非元素词
            non_elements = {'The', 'A', 'An', 'For', 'With', 'That', 'This', 'Is', 'Are', 'In', 'On', 'At'}
            elements = [e for e in elements if e not in non_elements]
            return elements
    
    @staticmethod
    def check_format_validity(text):
        """
        检查文本格式是否有效（是否包含 <material> 标签或有效的元素格式）
        返回: (is_valid, has_material_tag)
        """
        if pd.isna(text) or not text:
            return False, False
        
        text = str(text).strip()
        has_material_tag = bool(re.search(r'<material>.*?</material>', text, re.DOTALL | re.IGNORECASE))
        
        # 如果包含 <material> 标签，格式有效
        if has_material_tag:
            return True, True
        
        # 或者检查是否包含有效的化学元素格式
        element_pattern = r'\b[A-Z][a-z]?\b'
        elements = re.findall(element_pattern, text)
        # 过滤掉常见的非元素词
        non_elements = {'The', 'A', 'An', 'For', 'With', 'That', 'This', 'Is', 'Are', 'In', 'On', 'At'}
        valid_elements = [e for e in elements if e not in non_elements]
        
        return len(valid_elements) > 0, False
    
    @staticmethod
    def check_smact_validity(elements):
        """
        检查材料组成是否符合 SMACT 化学规则（如果 SMACT 可用）
        返回: (is_valid, formula)
        """
        if not SMACT_AVAILABLE or not elements:
            return None, None
        
        try:
            # 构建化学式
            counter = Counter(elements)
            formula = ''.join(f"{el}{cnt if cnt > 1 else ''}" for el, cnt in sorted(counter.items()))
            
            # 使用 SMACT 验证
            is_valid = smact_validity(formula)
            return is_valid, formula
        except Exception:
            return None, None
    
    @classmethod
    def process_composition_task(cls, task_entries, gt_materials=None):
        """
        处理组成到材料的生成任务，评估预测结果
        增强版：包含格式验证、SMACT 有效性检查、元素精度和新颖性检查
        """
        format_valid_flags = []
        smact_valid_flags = []
        composition_precisions = []
        novelty_flags = []
        
        if gt_materials is None:
            gt_materials = set()
        
        for index, entry in task_entries.iterrows():
            # 获取问题和预测
            question = entry.get("question", "")
            pred_text = entry.get("prediction", "")
            
            # 从问题中提取元素组成
            prompt_elements = extract_elements_from_prompt(question)
            
            # 提取预测中的材料组成
            pred_elements = cls.extract_material_composition(pred_text)
            
            # 检查格式有效性
            format_valid, has_material_tag = cls.check_format_validity(pred_text)
            format_valid_flags.append(format_valid)
            
            # 检查 SMACT 有效性（如果可用）
            smact_valid, formula = cls.check_smact_validity(pred_elements)
            if smact_valid is not None:
                smact_valid_flags.append(smact_valid)
            else:
                smact_valid_flags.append(None)
            
            # 计算组成精度（元素命中率）
            if prompt_elements:
                prec = composition_precision(prompt_elements, pred_text)
                composition_precisions.append(prec)
            else:
                composition_precisions.append(0.0)
            
            # 检查新颖性
            predicted_material = material_postprocessor(pred_text)
            if not predicted_material:
                predicted_material = pred_text.strip()
            
            is_novel = 0
            if predicted_material:
                if predicted_material not in gt_materials:
                    is_novel = 1
            novelty_flags.append(is_novel)
        
        result = {
            'format_valid_flags': format_valid_flags,
            'composition_precisions': composition_precisions,
            'novelty_flags': novelty_flags
        }
        
        # 只有当 SMACT 可用时才添加相关数据
        if SMACT_AVAILABLE:
            result['smact_valid_flags'] = smact_valid_flags
        
        return result
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        评估组成到材料生成任务的预测结果
        """
        data = load(eval_file)
        data = data[~pd.isna(data["prediction"])]
        assert 'question' in data and 'prediction' in data
        
        # 加载 ground truth materials（从训练数据）
        gt_materials = set()
        # 尝试从 eval_file 中推断训练数据路径
        if 'answer' in data.columns:
            for _, row in data.iterrows():
                answer = row.get('answer', '')
                mat = material_postprocessor(answer)
                if mat:
                    gt_materials.add(mat.strip())
        
        # 或者从原始数据集加载
        if not gt_materials:
            try:
                dataset_url = cls.DATASET_URL.get('composition2material', '')
                if osp.exists(dataset_url):
                    train_data = load(dataset_url)
                    if 'answer' in train_data.columns:
                        for _, row in train_data.iterrows():
                            answer = row.get('answer', '')
                            mat = material_postprocessor(answer)
                            if mat:
                                gt_materials.add(mat.strip())
            except Exception:
                pass
        
        # 获取dataset_name
        dataset_name = None
        for name in cls.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break
        
        # 根据dataset_name选用对应的eval函数
        if dataset_name in cls.material_generation_task:
            results = cls.process_composition_task(data, gt_materials)
            
            # 计算平均指标
            format_valid_flags = results['format_valid_flags']
            composition_precisions = results['composition_precisions']
            novelty_flags = results['novelty_flags']
            
            n = len(format_valid_flags)
            if n == 0:
                return {
                    "format_valid_ratio": 0.0,
                    "average_precision": 0.0,
                    "novel_material_ratio": 0.0
                }
            
            format_valid_ratio = sum(format_valid_flags) / n
            avg_precision = sum(composition_precisions) / n
            novelty_ratio = sum(novelty_flags) / n
            
            result_dict = {
                "total_samples": n,
                "format_valid_count": sum(format_valid_flags),
                "format_valid_ratio": format_valid_ratio,
                "average_precision": avg_precision,
                "average_precision_%": avg_precision * 100,
                "novel_material_count": sum(novelty_flags),
                "novel_material_ratio": novelty_ratio,
                "novel_material_ratio_%": novelty_ratio * 100
            }
            
            # 如果 SMACT 可用，添加 SMACT 相关指标
            if SMACT_AVAILABLE and 'smact_valid_flags' in results:
                smact_valid_flags = [f for f in results['smact_valid_flags'] if f is not None]
                if smact_valid_flags:
                    format_valid_count = sum(format_valid_flags)
                    smact_valid_count = sum(smact_valid_flags)
                    result_dict["smact_valid_count"] = smact_valid_count
                    result_dict["smact_validity_ratio_in_format_valid"] = (
                        smact_valid_count / format_valid_count if format_valid_count > 0 else 0.0
                    )
                    result_dict["smact_validity_ratio_in_format_valid_%"] = (
                        result_dict["smact_validity_ratio_in_format_valid"] * 100
                    )
                    result_dict["smact_validity_ratio_in_all"] = (
                        smact_valid_count / n if n > 0 else 0.0
                    )
                    result_dict["smact_validity_ratio_in_all_%"] = (
                        result_dict["smact_validity_ratio_in_all"] * 100
                    )
            
            return result_dict
        else:
            return {}

