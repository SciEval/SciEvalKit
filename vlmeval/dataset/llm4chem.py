import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from ..smp import *
from .text_base import TextBaseDataset
from .utils.llm4chem.utils.metrics import calculate_smiles_metrics, calculate_formula_metrics, calculate_text_metrics, calculate_number_metrics, calculate_boolean_metrics
from .utils.llm4chem.config import TASKS, TASK_TAGS, TASKS_WITH_SEMICOLON_REPLACE

# TODO: Now only support Top1 evaluation for generation tasks, need to support TopK evaluation in the future

# TODO: Modify the root path to https://opencompass.openxlab.space/ ... as needed
# TODO: See tos://tos-bjml-ai4scilab/scievalkit_benchmark/LLM4Chem/
DATASET_ROOT_PATH = "LLM4Chem/"

def extract_answer_part(outputs, left_tag, right_tag, mode='tag'):
    assert mode in ('tag', 'direct')

    assert isinstance(outputs, list)
    answers = []
    for text in outputs:
        if mode == 'direct' or (left_tag is None and right_tag is None):
            text = text.replace('<unk>', '').replace('</s>', '').strip()
            answers.append(text.strip())
            continue
        
        left_tag_pos = text.find(left_tag)
        if left_tag_pos == -1:
            answers.append('')
            continue
        right_tag_pos = text.find(right_tag)
        if right_tag_pos == -1:
            answers.append('')
            continue
        text = text[left_tag_pos + len(left_tag): right_tag_pos].strip()
        answers.append(text)
    return answers

def LLM4Chem_postprocess(text, task, *args, **kwargs):
    # delete content within <think> </think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    replace_semicolon = task in TASKS_WITH_SEMICOLON_REPLACE
    pred = extract_answer_part([text], *(TASK_TAGS[task]), mode='tag')[0]
    # task in TASKS_WITH_SEMICOLON_REPLACE needs semicolon replaced with a period
    if replace_semicolon:
        pred = pred.replace(';', '.')
    # no matched tag
    if pred == '':
        tag = TASK_TAGS[task][0]

        if (tag == '<BOOLEAN>'):
            # find the last yes/true/no/false in text, case insensitive
            ans = re.findall(r'\b(?:yes|true|no|false)\b', text, re.IGNORECASE)
            if ans:
                # if ans[-1] is yes/true
                if ans[-1].lower() in ('yes', 'true'):
                    return 'Yes'
                else:
                    return 'No'
            else:
                return ''

        if (tag == '<NUMBER>'):
            # find the last number in text
            # remove content within <SMILES> </SMILES> from text
            text_2 = re.sub(r'<SMILES>.*?</SMILES>', '', text, flags=re.DOTALL)
            ans = re.findall(r'-?\d*\.\d+|-?\d+', text_2)
            if ans:
                return ans[-1]
            else:
                return ''

        if (tag == '<MOLFORMULA>'):
            # find the last chemical formula in text
            ans = re.findall(r'[\[\(]?[A-Z][a-z]?\d*(?:\([A-Za-z0-9]+\)\d*)?[\]\)]?(?:[A-Z][a-z]?\d*|\([^\)]+\)\d*|\[[^\]]+\]\d*)*(?:[+-]{1,2})?(?:·\d*[A-Z][a-z]?\d*)*', text)
            if ans:
                return ans[-1]
            else:
                return ''
            
                
    # print(f"prediction: {pred}")
    return pred

# WARNING: You should ensure the Internet is connected while running this function for the first time.
def download_nltk():
    import nltk
    conda_prefix = os.environ.get('CONDA_PREFIX', None)
    if conda_prefix is not None:
        nltk_data_dir = os.path.join(conda_prefix, 'nltk_data')
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
        nltk.data.path.append(nltk_data_dir)
    else:
        nltk.download('wordnet', quiet=True)
        nltk_data_dir = '~/nltk_data'
    print(f"NLTK 'wordnet' downloaded to: {nltk_data_dir}")

class LLM4Chem(TextBaseDataset):
    TYPE = 'TEXT'

    DATASET_URL = {
        # TODO: DATASET_ROOT_PATH is a placeholder, need to modify to actual path or URL
        task: DATASET_ROOT_PATH + f"LLM4Chem_{task}.tsv" for task in TASKS
    }

    DATASET_MD5 = {
        task : None for task in TASKS
    }

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
        task = dataset_name
        answers = []
        predictions = []
        for index,entry in data.iterrows():
            answers.append(LLM4Chem_postprocess(str(entry['answer']), task))
            predictions.append(LLM4Chem_postprocess(str(entry['prediction']), task))
        
        answers = [[ans] for ans in answers]
        predictions = [[pred] for pred in predictions]
        # TODO: Now the following code support TopK evaluation, but from EvalKit framework, we cannot get K generated answers.

        pred_list = predictions
        gold_list = answers

        if (task == 'molecule_captioning'):
            download_nltk() # Ensure NLTK wordnet is downloaded

        if task in ('property_prediction-esol', 'property_prediction-lipo', 'property_prediction-bbbp', 'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider'):
            # set pred_list to [length * 1]
            pred_list = [[pred[0]] for pred in pred_list]

        if task in ('forward_synthesis', 'molecule_generation', 'name_conversion-i2s'):
            r = calculate_smiles_metrics(pred_list, gold_list)
        elif task in ('retrosynthesis', 'retrosynthesis_uspto50k'):
            r = calculate_smiles_metrics(pred_list, gold_list, metrics=('exact_match', 'fingerprint', 'multiple_match'))
        elif task in ('molecule_captioning',):
            r = calculate_text_metrics(
                pred_list,
                gold_list,
                text_model='allenai/scibert_scivocab_uncased',
                text_trunc_length=2048,
            )
        elif task in ('name_conversion-i2f', 'name_conversion-s2f'):
            r = calculate_formula_metrics(pred_list, gold_list, metrics=('element_match',))
        elif task in ('name_conversion-s2i',):
            r = calculate_formula_metrics(pred_list, gold_list, metrics=('split_match',))
        elif task in ('property_prediction-esol', 'property_prediction-lipo'):
            r = calculate_number_metrics(pred_list, gold_list)
        elif task in ('property_prediction-bbbp', 'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider'):
            r = calculate_boolean_metrics(pred_list, gold_list)
        else:
            raise ValueError(task)

        if 'num_t1_exact_match' in r and 'num_all' in r:
            r['top1_exact_match'] = round(r['num_t1_exact_match'] / r['num_all'] * 100, 2)
        if 'num_t5_exact_match' in r and 'num_all' in r:
            r['top5_exact_match'] = round(r['num_t5_exact_match'] / r['num_all'] * 100, 2)
        if 'num_t1_ele_match' in r and 'num_all' in r:
            r['top1_ele_match'] = round(r['num_t1_ele_match'] / r['num_all'] * 100, 2)
        if 'num_correct' in r and 'num_all' in r:
            r['accuracy'] = round(r['num_correct'] / r['num_all'] * 100, 2)
        if 'num_t1_split_match' in r and 'num_all' in r:
            r['top1_split_match'] = round(r['num_t1_split_match'] / r['num_all'] * 100, 2)
        if 'num_t5_split_match' in r and 'num_all' in r:
            r['top5_split_match'] = round(r['num_t5_split_match'] / r['num_all'] * 100, 2)

        return r