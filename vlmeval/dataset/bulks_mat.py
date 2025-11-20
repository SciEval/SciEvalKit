import warnings
import pandas as pd
import re
from abc import abstractmethod
from collections import Counter
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from ..smp import *
from .text_base import TextBaseDataset

# 尝试导入 SMACT 库（可选）
try:
    from smact.screening import smact_validity
    SMACT_AVAILABLE = True
except ImportError:
    SMACT_AVAILABLE = False
    warnings.warn("SMACT library not available. SMACT validity checks will be skipped.")

class BulkModulus2Material(TextBaseDataset):
    TYPE = 'TEXT'
    DATASET_URL = {
        'bulk_modulus2material': '/root/code/VLMEvalKit/LMUData/bulk_modulus2material_no_coords_train.tsv',
    }
    DATASET_MD5 = {
        'bulk_modulus2material': ''
    }  # MD5码暂时不需要，先置空
    material_generation_task = ['bulk_modulus2material']
    
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
    def normalize_composition(elements):
        """标准化元素列表，用于比较"""
        # 将元素列表转换为排序后的字符串（考虑重复）
        if not elements:
            return ""
        counter = Counter(elements)
        # 按元素字母排序，但保持计数
        sorted_items = sorted(counter.items())
        normalized = " ".join([f"{elem}×{count}" for elem, count in sorted_items])
        return normalized
    
    @staticmethod
    def compute_element_f1(pred_elements, gold_elements):
        """
        计算元素级别的 F1 分数
        """
        if not gold_elements:
            return 0.0, 0.0, 0.0
        
        pred_counter = Counter(pred_elements)
        gold_counter = Counter(gold_elements)
        
        # 计算交集（共同元素及其最小计数）
        intersection = sum((pred_counter & gold_counter).values())
        
        if intersection == 0:
            return 0.0, 0.0, 0.0
        
        pred_total = sum(pred_counter.values())
        gold_total = sum(gold_counter.values())
        
        precision = intersection / pred_total if pred_total > 0 else 0.0
        recall = intersection / gold_total if gold_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
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
    def process_material_generation_task(cls, task_entries):
        """
        处理材料生成任务，比较预测的材料组成和正确答案
        增强版：包含格式验证和 SMACT 有效性检查
        """
        exact_matches = []
        element_f1_scores = []
        element_precisions = []
        element_recalls = []
        format_valid_flags = []
        smact_valid_flags = []
        
        for index, entry in task_entries.iterrows():
            # 提取答案中的材料组成
            gold_text = entry["answer"]
            gold_elements = cls.extract_material_composition(gold_text)
            
            # 提取预测中的材料组成
            pred_text = entry["prediction"]
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
            
            # 计算精确匹配（标准化后的字符串匹配）
            gold_normalized = cls.normalize_composition(gold_elements)
            pred_normalized = cls.normalize_composition(pred_elements)
            exact_match = 1 if gold_normalized == pred_normalized else 0
            exact_matches.append(exact_match)
            
            # 计算元素级别的 F1 分数
            precision, recall, f1 = cls.compute_element_f1(pred_elements, gold_elements)
            element_precisions.append(precision)
            element_recalls.append(recall)
            element_f1_scores.append(f1)
        
        result = {
            'exact_matches': exact_matches,
            'element_f1_scores': element_f1_scores,
            'element_precisions': element_precisions,
            'element_recalls': element_recalls,
            'format_valid_flags': format_valid_flags
        }
        
        # 只有当 SMACT 可用时才添加相关数据
        if SMACT_AVAILABLE:
            result['smact_valid_flags'] = smact_valid_flags
        
        return result
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        评估材料生成任务的预测结果
        """
        data = load(eval_file)
        data = data[~pd.isna(data["prediction"])]
        assert 'answer' in data and 'prediction' in data
        
        # 获取dataset_name
        dataset_name = None
        for name in cls.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break
        
        # 根据dataset_name选用对应的eval函数
        if dataset_name in cls.material_generation_task:
            results = cls.process_material_generation_task(data)
            
            # 计算平均指标
            exact_matches = results['exact_matches']
            element_f1_scores = results['element_f1_scores']
            element_precisions = results['element_precisions']
            element_recalls = results['element_recalls']
            format_valid_flags = results['format_valid_flags']
            
            n = len(exact_matches)
            if n == 0:
                return {
                    "exact_match_accuracy": 0.0,
                    "element_f1": 0.0,
                    "element_precision": 0.0,
                    "element_recall": 0.0,
                    "format_valid_ratio": 0.0
                }
            
            exact_match_acc = sum(exact_matches) / n
            avg_element_f1 = sum(element_f1_scores) / n
            avg_element_precision = sum(element_precisions) / n
            avg_element_recall = sum(element_recalls) / n
            format_valid_ratio = sum(format_valid_flags) / n
            
            result_dict = {
                "exact_match_accuracy": exact_match_acc,
                "element_f1": avg_element_f1,
                "element_precision": avg_element_precision,
                "element_recall": avg_element_recall,
                "format_valid_ratio": format_valid_ratio
            }
            
            # 如果 SMACT 可用，添加 SMACT 相关指标
            if SMACT_AVAILABLE and 'smact_valid_flags' in results:
                smact_valid_flags = [f for f in results['smact_valid_flags'] if f is not None]
                if smact_valid_flags:
                    format_valid_count = sum(format_valid_flags)
                    smact_valid_count = sum(smact_valid_flags)
                    result_dict["smact_valid_count"] = smact_valid_count
                    result_dict["format_valid_count"] = format_valid_count
                    result_dict["smact_validity_ratio_in_format_valid"] = (
                        smact_valid_count / format_valid_count if format_valid_count > 0 else 0.0
                    )
                    result_dict["smact_validity_ratio_in_all"] = (
                        smact_valid_count / n if n > 0 else 0.0
                    )
            
            return result_dict
        else:
            return {} 