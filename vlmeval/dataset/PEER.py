import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from ..smp import *
from .text_base import TextBaseDataset
from .utils.PEER.peer import *    #导入辅助函数
import numpy as np
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_score, recall_score, f1_score
from os import environ
import requests
from openai import OpenAI
MAX_RETRIES = 3
BACKOFF_SEC = 2

def _retry_api(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = fn(*args, **kwargs)
            if result is not None:
                return result
            raise ValueError("Received None")
        except Exception as e:
            last_exc = e
            sleep_time = BACKOFF_SEC ** attempt
            print(f"[retry] attempt {attempt} failed ({e}), retrying in {sleep_time}s…")
            time.sleep(sleep_time)
    raise last_exc

def ask_gpt25(client, gpt_model, question, answer, prediction):

    prompt = (
        # "请判断这个回答是否正确。 定义：“正确”：模型回答的核心结论（如是否存在相互作用）与参考答案完全一致（不要求字面相同）；“错误”：模型回答的核心结论与参考答案相反，或未明确表达核心结论。"
        # f"参考答案：{answer}"
        # f"模型回答：{prediction}"
        # "如果正确，请回答'True'；如果错误，请回答'False'。请只回答'True'或'False'。"

        "Please determine whether this answer is correct. Definition: 'Correct': The core conclusion of the model's answer (if any) is completely consistent with the reference answer (literal identity is not required). 'Incorrect': The core conclusion of the model's answer is consistent with the reference answer, or the core conclusion is not clearly expressed."
        f"Reference answer: {answer}"
    f"Model answer: {prediction}"
    "If correct, answer 'True'; if incorrect, answer 'False'. Please only answer 'True' or 'False'."
    )

    def _call():
        payload = dict(
            model=gpt_model,
            messages=[
                {
                    "role": "user",
                    "content": [{'type': 'text', 'text':prompt}]
                }
            ],
            temperature=0
        )
        key = os.environ['OPENAI_API_KEY']

        response = requests.post(
            os.environ['OPENAI_API_BASE'],
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'},
            data=json.dumps(payload),
        )

        resp_struct = json.loads(response.text)
        result = resp_struct['choices'][0]['message']['content'].strip().upper()
        print("=== GPT 判断结果 ===")
        print(f"Prompt:\n{prompt}")
        print(f"Output:\n{result}")
        return result

    try:
        return _retry_api(_call)
    except Exception as e:
        print(f"[GPT ERROR] Exception: {e}")
        return ''

def ask_gpt25_batch(client, gpt_model, questions, answers, predictions):
    results = [None] * len(questions)

    def task(index, client, gpt_model, q, a, p):
        try:
            result = ask_gpt25(client, gpt_model, q, a, p)
            results[index] = result
        except Exception as e:
            results[index] = ''
            print(f"[GPT ERROR] 批次样本 {index} 出错: {e}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(task, i, client, gpt_model, q, a, p)
            for i, (q, a, p) in enumerate(zip(questions, answers, predictions))
        ]
        for future in as_completed(futures):
            pass

    return results

class PEER(TextBaseDataset):
    TYPE = 'TEXT'
    tsv_root = '/mnt/shared-storage-user/ai4sreason/wangyizhou/Evalkit_tools/tsv_files'
    DATASET_URL = {
        'solubility': f'{tsv_root}/solubility.tsv',
        'stability': f'{tsv_root}/stability.tsv',
        'human_ppi': f'{tsv_root}/human_ppi.tsv',
        'yeast_ppi': f'{tsv_root}/yeast_ppi.tsv',
    }
    DATASET_MD5 = {
        'solubility': None,
        'stability': None,
        'human_ppi': None,
        'yeast_ppi': None

    }#MD5码暂时不需要，先置空


    protein_tasks = [
        'solubility',
        'stability',
        'human_ppi',
        'yeast_ppi'
    ]

    gpt_model = 'gpt-4o-mini-2024-07-18'

    client = OpenAI()
   
    def process_protein_tasks(data, dataset_name):
        result_values = []
        label_values = []
        original_predictions = []
        original_references = []
        task_processed_data = []
        num_all = len(data)
        
        def remove_think_tag(text):
            text = text.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
            match = re.search(r'The answer is\s+(Yes|No)', text, re.IGNORECASE)
            if match:
                return match.group(1)

            positive_patterns = [
                r'will be soluble',
                r'will dissolve',
                r'is soluble',
                r'can be predicted',
                r'positive',
                r'Yes',
                r'correct',
                r'valid',
                r'accurate',
                r'certainly',
                r'indeed',
                r'affirmative',
                r'highly soluble',
                r'easily soluble',
                r'dissolves easily',
                r'is assured',
                r'be soluble'
            ]

            negative_patterns = [
                r'will not be soluble',
                r'is not soluble',
                r'will not dissolve',
                r'low solubility',
                r'low',
                r'cannot be predicted',
                r'negative',
                r'No',
                r'incorrect',
                r'invalid',
                r'inaccurate',
                r'impossible',
                r'not possible',
                r'denied',
                r'be insoluble',
            ]

            for pattern in positive_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return "Yes"

            for pattern in negative_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return "No"

            return ""
        
        def float_compare(text, compare_number=1.):
            if not isinstance(text, str):
                return ""
            try:
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
                # 提取文本中的数字
                match = re.search(r'[-+]?\d*\.\d+|\d+', text)
                if match:
                    value = float(match.group(0))
                    # 比较数值
                    if value > compare_number:
                        return "Yes"
                    else:
                        return "No"
                else:
                    # 如果没有找到数字，返回空字符串
                    return ""
            except ValueError:
                # 如果转换失败，返回空字符串
                return ""

        for index, entry in data.iterrows():
            
            model_output = str(entry['prediction'])
            label = str(entry['answer'])
            original_predictions.append(model_output)
            original_references.append(label)
            if dataset_name == 'stability':
                label = float_compare(label)
                model_output = float_compare(model_output)
            else:
                label = remove_think_tag(label)
                model_output = remove_think_tag(model_output)

            result_values.append([model_output])
            label_values.append([label])
        
        predictions = result_values
        references = label_values
        
        return references, predictions, original_references, original_predictions
    

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        result_values = []
        label_values = []
        data = load(eval_file)
        data= data[~pd.isna(data["prediction"])]
        assert 'answer' in data and 'prediction' in data 
        dataset_name = None

        for name in self.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break
        
        postprocessed_references, postprocessed_predictions, references, predictions = self.process_protein_tasks(data, dataset_name)

        num_all = len(postprocessed_predictions)
        num_correct, num_no_answer, num_invalid = 0, 0, 0
        num_gpt_called = 0
        new_pred, new_gold = [], []

        to_recheck_indices = []
        to_recheck_golds = []
        to_recheck_preds = []
        for i, (pred_item, gold_item) in enumerate(zip(postprocessed_predictions, postprocessed_references)):
            pred = pred_item[0]
            gold = gold_item[0]

            if pred not in ('yes', 'no'):
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])
                continue

            if pred == 'yes':
                pred_bin = 1
            elif pred == 'no':
                pred_bin = 0
            else:
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])
                continue

            if gold == 'yes':
                gold_bin = 1
            elif gold == 'no':
                gold_bin = 0
            else:
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])
                continue


            if pred_bin == gold_bin:
                num_correct += 1
                # import pdb; pdb.set_trace()
                print(references[i][0],'\n',predictions[i][0],'----')
                new_pred.append(pred_bin)
                new_gold.append(gold_bin)
            else:
                to_recheck_indices.append(i)
                to_recheck_golds.append(references[i][0])
                to_recheck_preds.append(predictions[i][0])


        if to_recheck_indices:
            rechecked_preds = ask_gpt25_batch(
                self.client,
                self.gpt_model,
                ["" for _ in to_recheck_indices],
                to_recheck_golds,
                to_recheck_preds
            )
            num_gpt_called += len(rechecked_preds)

            for i, result in enumerate(rechecked_preds):
                result = result.strip().lower()
                if 'true' in result:
                    num_correct += 1
                    pred_bin = 1
                    gold_bin = 1
                elif 'false' in result:
                    pred_bin = 0
                    gold_bin = 1
                else:
                    pred_bin = 1
                    gold_bin = 0

                new_pred.append(pred_bin)
                new_gold.append(gold_bin)

        new_pred = np.array(new_pred)
        new_gold = np.array(new_gold)

        results = {
            'num_all': num_all,
            'num_correct': num_correct,
            'num_no_answer': num_no_answer,
            'num_invalid': num_invalid,
            'num_gpt_called': num_gpt_called,
            'accuracy': num_correct / num_all * 100,
            'acc_wo_no_answer_invalid': num_correct / (num_all - num_no_answer - num_invalid) * 100
            if (num_all - num_no_answer - num_invalid) > 0 else 0,
            'precision': precision_score(new_gold, new_pred, zero_division=0) * 100,
            'recall': recall_score(new_gold, new_pred, zero_division=0) * 100,
            'f1_score': f1_score(new_gold, new_pred, zero_division=0) * 100,
        }
        return  results