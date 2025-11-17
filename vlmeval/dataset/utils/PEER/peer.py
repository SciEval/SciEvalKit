# dataset: PEER
# task : solubility prediction

import json
import multiprocessing
from functools import partial
import re
from os import environ
from typing import Union

import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem, RDLogger
from tqdm import tqdm


from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
import numpy as np


def calculate_accuracy(pred_text_list, gold_text_list):
    assert len(pred_text_list) == len(gold_text_list)
    num_all = len(pred_text_list)
    metrics = {}
    metrics['num_all'] = num_all
    num_no_answer = 0
    num_invalid = 0
    num_correct = 0
    new_pred_text_list, new_gold_text_list = [], []
    for (pred_item, gold_item) in zip(pred_text_list, gold_text_list):
        if pred_item is None or pred_item == '':
            num_no_answer += 1
            continue
        assert len(pred_item) == 1
        assert len(gold_item) == 1
        pred_item = pred_item[0].strip().lower()
        gold_item = gold_item[0].strip().lower()
        if pred_item == '':
            num_no_answer += 1
            continue
        if pred_item not in ('yes', 'no'):
            num_invalid += 1
            continue
        pred_item = 1 if pred_item == 'yes' else 0
        gold_item = 1 if gold_item == 'yes' else 0
        new_pred_text_list.append(pred_item)
        new_gold_text_list.append(gold_item)
        if gold_item == pred_item:
            num_correct += 1

    metrics['num_no_answer'] = num_no_answer
    metrics['num_invalid'] = num_invalid
    metrics['num_correct'] = num_correct

    # return metrics

    new_gold_text_list = np.array(new_gold_text_list)
    new_pred_text_list = np.array(new_pred_text_list)

    # macro_roc_auc_score = roc_auc_score(new_gold_text_list, new_pred_text_list)
    f1 = f1_score(new_gold_text_list, new_pred_text_list)
    # metrics['roc_auc_score'] = macro_roc_auc_score
    metrics['accuracy'] = num_correct / (num_all) * 100
    metrics['acc_wo_no_answer_invalid'] = num_correct / (num_all - num_no_answer - num_invalid) * 100 if (num_all - num_no_answer - num_invalid) > 0 else 0
    metrics['precision'] = precision_score(new_gold_text_list, new_pred_text_list) * 100
    metrics['recall'] = recall_score(new_gold_text_list, new_pred_text_list) * 100
    metrics['f1_score'] = f1* 100

    return metrics






