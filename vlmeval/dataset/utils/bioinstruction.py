import re
from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import json
import os
from datasets import Dataset, DatasetDict
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from transformers import pipeline
from tqdm import tqdm
import torch
import ast
import random
import sys

RNA_CLASSES = sorted(['5S_rRNA', '5_8S_rRNA', 'tRNA', 'ribozyme', 'CD-box', 'miRNA', 'Intron_gpI', 'Intron_gpII', 'HACA-box', 'riboswitch', 'IRES', 'leader', 'scaRNA'], key=len, reverse=True)

modification_classes = ['AtoI','m6Am','m1A', 'm5C', 'm5U', 'm6A',  'm7G', 'Psi', 'Am', 'Cm', 'Gm', 'Um','none']


def classify_by_keywords(text):
    positive_keywords = ["Yes",'yes',"positive","Positive","empirical","plausible","confirms","have detected","are discernible","are supported","is supported","display","detected the presence",
     "shows evidence","has been identified","shows","has identified","contains ","exhibits evidence","is plausible","contains identifiable","Indeed","reveals the presence","include","are present","definitely has","soluble","displays regions","has a high solubility","dissolves easily","Solubility is expected","is expected to dissolve","is predicted","is likely","is expected","is expected to dissolve","will dissolve","dissolves easily"]

    negative_keywords = ["No",'no',"negative","Negative","insoluble","does not","unlikely",'absence', 'not found', 'not detected', 'not associated', 'not inferred', 'not linked', 'does not indicate', 'no evidence', 'not predicted', 'absent',"not present","no indicators","not exhibit","are absent","found none","did not reveal","lacks","exhibits no","insolubility","low solubility","not soluble","not be soluble","does not display regions"]
    
    dont_know_keywords = ['don\'t know', 'unknown', 'unsure', 'uncertain', 'not applicable',"cannot confirm"]

    text_lower = text.lower()

    # 为了安全，转义关键词中的特殊字符，并用'|'（或）连接
    # \b确保匹配的是整个单词
    negative_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in negative_keywords) + r')\b'
    positive_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in positive_keywords) + r')\b'
    dont_know_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in dont_know_keywords) + r')\b'
    
    # 1. 检查负面关键词
    if re.search(negative_pattern, text_lower):
        return 0
    # 2. 检查正面关键词
    elif re.search(positive_pattern, text_lower):
        return 1
    # 3. 检查 "不知道" 关键词
    elif re.search(dont_know_pattern, text_lower):
        return "dont_know"
    else:
        return None


def classify_by_sentiment_model(text):
    classifier  = pipeline("zero-shot-classification", model="/scievalkit_benchmark/Bioinstruction/bart_large_mnli")


    text = [str(t).replace("</s>", "").replace("<pad>", "").strip() for t in text]
    candidate_labels = ['Yes,I can positively identify', 'No,My answer is negative',"This protein is expected to dissolve in water","This protein is not expected to dissolve in water"]
    outputs=classifier(text, candidate_labels,batch_size=64)
    processed_results = []
    for output in outputs:
        top_label = output['labels'][0]
        top_score = output['scores'][0]
        
        if top_label == 'Yes,I can positively identify'or top_label=="This protein is expected to dissolve in water":
            result_class = 1  # 肯定
        else:
            result_class = 0  # 否定
        processed_results.append((result_class, top_score))
    print("processed_results",processed_results)
    return processed_results

def extract_modifications(text):
    extracted_modifications = []
    for mod_class in modification_classes:
        # Use word boundaries to ensure whole-word match
        if re.search(rf'\b{mod_class}\b', text):
            extracted_modifications.append(mod_class)
    return extracted_modifications

def convert_to_binary_vector(modifications, classes=modification_classes):
    binary_vector = []
    
    # Handle case where modifications is None
    if modifications is None:
        modifications = []  # Treat None as an empty list
    
    for mod in classes:
        if mod in modifications:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector

def extract_numeric_values(text):
    text = text.replace("5'", "five'")
    text=text.replace("3'","three")
    matches = re.findall(r'(?<![a-zA-Z])[-‑]?\d+\.?\d*', str(text))
    numeric_values = []
    for num in matches:
        num = num.replace('‑', '-')
        value = np.float64(num)  

        if value.is_integer():
            value = f'{int(value):.6g}'
        else:  
            value = f'{value:.6g}'
        
        numeric_values.append(float(value))  

    return numeric_values

def extract_rna_family(text):
    for rna_class in RNA_CLASSES:
        if rna_class in text:  
            return rna_class
    return None
    
def compute_mixed_score(label_values, result_values, threshold=300, max_value=1000):
    if len(result_values) == 0:
        return {
            "mixed_score": "Error: Empty data."
        }
    elif len(result_values) != len(label_values):
        return {
            "mixed_score": "Error: Mismatch in the number of extracted numeric values"
        }
    # Convert the label and result values to numeric arrays using pandas to handle non-numeric entries
    result_values = pd.to_numeric(result_values, errors='coerce').flatten()
    label_values = pd.to_numeric(label_values, errors='coerce').flatten()
    # Identify near-infinity values
    near_infinity_mask = np.abs(result_values) > max_value
    if near_infinity_mask.any():

        print(f"Warning: Found {sum(near_infinity_mask)} result values too large will be assigned a mixed score of 0. Large result values: {result_values[near_infinity_mask]} ")
    
    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Assign a mixed score of 0 to near-infinity pairs
    num_infinity_values = near_infinity_mask.sum()
    if num_infinity_values > 0:
        mixed_score_infinity = 0

    # Convert to binary based on the threshold for valid values
    label_binary = (valid_label_values < threshold).astype(int)
    result_binary = (valid_result_values < threshold).astype(int)
    
    # Compute precision, recall, F1 score for valid values
    precision = precision_score(label_binary, result_binary, average='binary')
    recall = recall_score(label_binary, result_binary, average="binary")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    try:
        # Compute mean absolute error (MAE) for valid values
        mae = mean_absolute_error(valid_label_values, valid_result_values)

    except ValueError as e:

        mae = np.inf  # Fallback to infinity if error occurs
    
    # Mask to keep only values in the range [0, threshold] for valid values
    mask = (valid_result_values >= 0) & (valid_result_values <= threshold)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(valid_label_values[mask], valid_result_values[mask])
    else:
        range_mae = 100  # Fallback if no values within the range

    # Ensure MAE and range_mae are within reasonable bounds to avoid overflow
    mae = min(mae, 100)
    range_mae = min(range_mae, 100)

    # Compute mixed score for valid values
    mixed_score_valid = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    print(f"(1 - mae / 100) * 0.5={(1 - mae / 100) * 0.5}\n (1 - range_mae / 100)={(1 - range_mae / 100)}\n (1 - range_mae / 100) * f1 * 0.5={(1 - range_mae / 100) * f1 * 0.5}")

    # Compute the final mixed score, averaging in the score for the near-infinity pairs
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    
    if num_infinity_values > 0:
        final_mixed_score = (mixed_score_valid * total_valid_points + mixed_score_infinity * num_infinity_values) / total_data_points
    else:
        final_mixed_score = mixed_score_valid  # Edge case: no near-infinity values

    return  {"mixed_score":final_mixed_score}

# Compute matthews correlation coefficient (MCC)
def compute_MCC(label_classes, result_classes):
    if len(result_classes) == 0:
        return {
            "MCC": "Error: Empty data."
        }
    elif len(result_classes) != len(label_classes):
        return {
            "MCC": "Error: Mismatch in the number of extracted numeric values."
        }
    else:
        mcc = matthews_corrcoef(label_classes, result_classes)
        return {
            "MCC": mcc
        }


def compute_R2(label_values, result_values):
    # from sklearn.metrics import r2_score

    y_true = np.asarray(label_values, dtype=float).flatten()

    y_pred = np.asarray(result_values, dtype=float).flatten()

    if len(result_values) == 0:
        return {
            "R2": "Error: Empty data."
        }
    
    # Check for equal length of arrays
    elif len(result_values) != len(label_values):
        return {
            "R2": "Error: Mismatch in the number of extracted numeric values."
        }

   # Convert the label and result values to numpy arrays   
    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)
    
    
    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Pearson correlation coefficient for valid values
    if len(valid_result_values) > 0:
        try:
            pcc, _ = pearsonr(valid_label_values, valid_result_values)
            R2 = pcc ** 2
        except Exception as e:

            R2 = np.inf  # Fallback to inf if computation fails
    else:
        R2 = 0  # Fallback if no valid pairs

    # Combine R2 score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_R2_score = (R2 * total_valid_points + 0 * num_infinity_values) / total_data_points
    else:
        final_R2_score = R2  # Edge case: no near-infinity values

    return {
        "R2": final_R2_score
    }


def compute_Acc(label_classes, result_classes):
    if len(result_classes) == 0:
        return {
            "Acc": "Error: Insufficient data for classification. Number of model outputs is 0."
        }
    elif len(result_classes) != len(label_classes):
        return {
            "Acc": "Error: Mismatched labels. The number of model outputs does not match the number of labels."
        }
    else:
        acc = accuracy_score(label_classes, result_classes)
        return {
            "Acc": acc
        }

def compute_spearman(label_values, result_values):
    if len(result_values) == 0:
        return {
            "spearman": "Error: Empty data"
        }
    elif len(result_values) != len(label_values):
        return {
            "spearman": "Error: Mismatch in the number of extracted numeric values"
        }
    result_values = np.array(result_values).flatten()
    label_values = np.array(label_values).flatten()

    # Identify explicitly assigned infinity values
    near_infinity_mask = np.isinf(result_values)
    
    
    # Exclude near-infinity pairs from the main calculation
    valid_mask = ~near_infinity_mask & np.isfinite(result_values) & np.isfinite(label_values)
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    # Compute Spearman correlation for valid values
    if len(valid_result_values) > 0:
        spearman, _ = spearmanr(valid_label_values, valid_result_values)
    else:
        spearman = 0  # Fallback if no valid pairs

    # Combine Spearman score for valid and infinity values
    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        final_spearman_score = (spearman * total_valid_points + 0 * num_infinity_values) / total_data_points
    else:
        final_spearman_score = spearman  

    return {
        "spearman": final_spearman_score
    }

def compute_R2_for_ProgrammableRNASwitches_task(task_entries):
    
    on_result_values = []
    off_result_values = []
    on_off_result_values = []

    on_label_values = []
    off_label_values = []
    on_off_label_values = []

    over_len=0
    miss_len=0
    num_all=len(task_entries)
    for index,entry in task_entries.iterrows():
        label = entry["category"]
        label = ast.literal_eval(label)
        on_label = float(label["ON"])
        off_label = float(label["OFF"])
        on_off_label = float(label["ON_OFF"])
        if "<summary>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("<summary>")[-1]
        # Extract numeric values from the model output
        if "</think>" in entry["prediction"]:
            entry["prediction"]=entry["prediction"].split("</think>")[-1]
        else:
            if "<think>" in entry["prediction"]:
                over_len+=1
            else:
                miss_len+=1
        extracted_result = extract_numeric_values(entry["prediction"])

        # Handle missing or invalid data by assigning np.nan
        if len(extracted_result) != 3:
            on_result_values.append(np.nan)
            off_result_values.append(np.nan)
            on_off_result_values.append(np.nan)
        else:
            on_result = extracted_result[0]
            off_result = extracted_result[1]
            on_off_result = extracted_result[2]
            on_result_values.append(on_result)
            off_result_values.append(off_result)
            on_off_result_values.append(on_off_result)

        # Append the label values
        on_label_values.append(on_label)
        off_label_values.append(off_label)
        on_off_label_values.append(on_off_label)


    # Convert to numpy arrays for easier manipulation
    on_result_values = np.array(on_result_values)
    off_result_values = np.array(off_result_values)
    on_off_result_values = np.array(on_off_result_values)

    on_label_values = np.array(on_label_values)
    off_label_values = np.array(off_label_values)
    on_off_label_values = np.array(on_off_label_values)

    # Filter out NaN values in ON, OFF, and ON/OFF result/label pairs
    on_valid_mask = np.isfinite(on_result_values) & np.isfinite(on_label_values)
    off_valid_mask = np.isfinite(off_result_values) & np.isfinite(off_label_values)
    on_off_valid_mask = np.isfinite(on_off_result_values) & np.isfinite(on_off_label_values)



    # Filter the valid ON, OFF, and ON/OFF values
    on_result_values = on_result_values[on_valid_mask]
    off_result_values = off_result_values[off_valid_mask]
    on_off_result_values = on_off_result_values[on_off_valid_mask]

    on_label_values = on_label_values[on_valid_mask]
    off_label_values = off_label_values[off_valid_mask]
    on_off_label_values = on_off_label_values[on_off_valid_mask]

    # Compute R2 for valid ON, OFF, and ON/OFF values
    try:
        on_R2 = compute_R2(on_result_values, on_label_values)['R2'] if len(on_result_values) > 0 else 0
    except Exception as e:

        on_R2 = 0  # Assign 0 in case of error

    try:
        off_R2 = compute_R2(off_result_values, off_label_values)['R2'] if len(off_result_values) > 0 else 0
    except Exception as e:

        off_R2 = 0  # Assign 0 in case of error

    try:
        on_off_R2 = compute_R2(on_off_result_values, on_off_label_values)['R2'] if len(on_off_result_values) > 0 else 0
    except Exception as e:
    
        on_off_R2 = 0  # Assign 0 in case of error

    # Combine R2 scores for ON, OFF, and ON/OFF values
    total_on_points = max(len(on_result_values) + np.sum(~on_valid_mask), 1)
    total_off_points = max(len(off_result_values) + np.sum(~off_valid_mask), 1)
    total_on_off_points = max(len(on_off_result_values) + np.sum(~on_off_valid_mask), 1)

    # Assign average R2 with 0 for invalid entries
    final_on_R2 = (on_R2 * len(on_result_values)) / total_on_points if len(on_result_values) > 0 else 0
    final_off_R2 = (off_R2 * len(off_result_values)) / total_off_points if len(off_result_values) > 0 else 0
    final_on_off_R2 = (on_off_R2 * len(on_off_result_values)) / total_on_off_points if len(on_off_result_values) > 0 else 0

    avg_R2 = (final_on_R2 + final_off_R2 + final_on_off_R2) / 3

    return {
        "R2": avg_R2
    }

def compute_PCC_for_enhancer_activity_task( task_entries):
    hk_result_values = []
    dev_result_values = []
    
    hk_label_values = []
    dev_label_values = []


    over_len=0
    miss_len=0
    num_all=len(task_entries)
    # Loop through each entry in the task
    for index,entry in task_entries.iterrows():
        label = entry["category"]
        label = ast.literal_eval(label)
        if "<summary>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("<summary>")[-1]
        if "</think>" in entry["prediction"]:
            entry["prediction"]=entry["prediction"].split("</think>")[-1]
        else:
            if "<think>" in entry["prediction"]:
                over_len+=1
            else:
                miss_len+=1
        model_output = entry["prediction"]
        hk_label = float(label["hk"])
        dev_label = float(label["dev"])

        # Extract model output values for HK and Dev enhancer activity
        extracted_result = extract_numeric_values(model_output)
        
        # Handle missing or invalid data by assigning np.inf
        if len(extracted_result) != 2:
      
            hk_result_values.append(np.inf)
            dev_result_values.append(np.inf)
        else:
            hk_result = extracted_result[0]
            dev_result = extracted_result[1]
            hk_result_values.append(hk_result)
            dev_result_values.append(dev_result)

        # Append the label values
        hk_label_values.append(hk_label)
        dev_label_values.append(dev_label)


    # Convert to numpy arrays for easier manipulation
    hk_result_values = np.array(hk_result_values)
    dev_result_values = np.array(dev_result_values)
    hk_label_values = np.array(hk_label_values)
    dev_label_values = np.array(dev_label_values)

    # Filter out NaN or inf values in both HK and Dev result/label pairs
    hk_valid_mask = np.isfinite(hk_result_values) & np.isfinite(hk_label_values)
    dev_valid_mask = np.isfinite(dev_result_values) & np.isfinite(dev_label_values)


    # Filter the valid HK and Dev values
    hk_result_values = hk_result_values[hk_valid_mask]
    hk_label_values = hk_label_values[hk_valid_mask]
    dev_result_values = dev_result_values[dev_valid_mask]
    dev_label_values = dev_label_values[dev_valid_mask]

    # Compute Pearson correlation for valid HK and Dev enhancer activities
    if len(hk_result_values) > 0:
        try:
            hk_pcc, _ = pearsonr(hk_result_values, hk_label_values)
        except Exception as e:
 
            hk_pcc = np.inf  # Set to inf in case of errors
    else:
        return {
            "PCC": "Error: HK has insufficient valid data after removing NaNs and infs."
        }
    if len(dev_result_values) > 0:
        try:
            dev_pcc, _ = pearsonr(dev_result_values, dev_label_values)
        except Exception as e:
     
            dev_pcc = np.inf  # Set to inf in case of errors
    else:
        return {
            "PCC": "Error: Dev has insufficient valid data after removing NaNs and infs."
        }
    # Combine results with NaN/inf values consideration
    total_hk_points = len(hk_result_values) + np.sum(~hk_valid_mask)
    total_dev_points = len(dev_result_values) + np.sum(~dev_valid_mask)

    # Assign mixed score with 0 for invalid entries
    final_hk_pcc = (hk_pcc * len(hk_result_values) + 0 * np.sum(~hk_valid_mask)) / total_hk_points if len(hk_result_values) > 0 else 0
    final_dev_pcc = (dev_pcc * len(dev_result_values) + 0 * np.sum(~dev_valid_mask)) / total_dev_points if len(dev_result_values) > 0 else 0

    return {
        "PCC": {
            "hk_PCC": final_hk_pcc,
            "dev_PCC": final_dev_pcc                
        }
    }
def ec_to_multihot(ec_list, ec_labels):
    multihot = torch.zeros(len(ec_labels))
    if not ec_list:  # Check if ec_list is empty
        return multihot
    multihot = torch.zeros(len(ec_labels))
    for ec in ec_list:
        if ec in ec_labels:
            idx = ec_labels.index(ec)
            multihot[idx] = 1
    return multihot


def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold. 
    Handles cases where either predictions or targets are empty.
    
    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
        
    Returns:
        float: The maximum F1 score or 0.0 if inputs are empty.
    """
    # Check if either pred or target is empty
    if pred.numel() == 0 or target.numel() == 0:

        return 0.0

    # Proceed with the original logic if inputs are not empty
    order = pred.argsort(descending=True, dim=1, stable=True)
    # print(f"order: {order}")
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    # print("isstart {}".format(is_start))
    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()
    
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    
    if torch.isnan(all_f1).any():

        return 0.0

    return all_f1.max()

def compute_Fmax_for_FunctionEC_task(task_entries, ec_labels):
    all_preds = []
    all_labels = []

    over_len=0
    miss_len=0
    num_all=len(task_entries)
    for index,entry in task_entries.iterrows():
        if "<summary>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("<summary>")[-1]
        if "</think>" in entry['prediction']:
            entry['prediction']=entry['prediction'].split("</think>")[-1]
        else:
            if "<think>" in entry["prediction"]:
                over_len+=1
            else:
                miss_len+=1
        if "<summary>" in entry['prediction']:
            entry['prediction']=entry['prediction'].split("<summary>")[-1]  
        # Parse the EC numbers from 'output' and 'label'
        label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', entry['category'])
        result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', str(entry['prediction']))

        # Convert EC numbers to multi-hot vectors
        pred_multihot = ec_to_multihot(result_ec, ec_labels) 
        label_multihot = ec_to_multihot(label_ec, ec_labels)

        # Store the results
        all_preds.append(pred_multihot)
        all_labels.append(label_multihot)
    

    # # Stack the predictions and targets for batch processing
    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)

    # Compute the Fmax score
    try:
        fmax_score = count_f1_max(all_preds, all_labels)
    except ValueError as e:

        fmax_score = None

    return {
        "Fmax": fmax_score.item()
    }



def compute_AUC_for_Modification_task(task_entries):
    y_true = []
    y_pred = []
    over_len=0
    miss_len=0
    num_all=len(task_entries)
    for index,entry in task_entries.iterrows():
        #MARK:gaile
        if "<summary>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("<summary>")[-1]
        if "</think>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("</think>")[-1]
        else:
            if "<think>" in entry["prediction"]:
                over_len+=1
            else:
                miss_len+=1
        predicted_modifications = extract_modifications(entry["prediction"])
        if len(predicted_modifications)!=1:
            predicted_modifications= ['none'] 

        #print(predicted_modifications)
        true_modifications = entry["category"].split(',')
           
        # Handle case where result is empty and label is "none"
        if not predicted_modifications :
            # Classify by keyword
            predicted_modifications = classify_by_keywords(entry["prediction"])

            # If keyword negative, assigned to prediction to be the "none" class
            if predicted_modifications == None:
                predicted_modifications = ['none']

            elif predicted_modifications == 1:
                predicted_modifications = []
            
            # If the result cannot be classified, use the sentiment model
            elif predicted_modifications is None:
                sentiment_result, sentiment_score = classify_by_sentiment_model(entry["prediction"])
                # If classified as negative, manually label as 'none'
                if sentiment_result == 0:
                    predicted_modifications = ['none'] 
                    
                else:
                    predicted_modifications = []
        
        # Convert the predicted and true modifications to binary vectors
        y_true.append(convert_to_binary_vector(true_modifications))
        y_pred.append(convert_to_binary_vector(predicted_modifications))
        
    # Compute the AUC for each class, then average the AUC across all classes
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError as e:
        auc = None
    return {
        "AUC": auc
    }

def compute_Acc_for_NoncodingRNAFamily_task(task_entries):
    correct_count = 0
    total_count = 0
    over_len=0
    miss_len=0
    num_all=len(task_entries)
    for index,entry in task_entries.iterrows():
        if "<summary>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("<summary>")[-1]
        if "</think>" in entry["prediction"]:
            entry["prediction"] = entry["prediction"].split("</think>")[-1]
            result_family = extract_rna_family(entry["prediction"]) 
        else:
            if "<think>" in entry["prediction"]:
                over_len+=1
            else:
                miss_len+=1
            #result_family = "None"
            result_family = extract_rna_family(entry["prediction"]) 

        label_family = entry["category"]

        if result_family == label_family:
            correct_count += 1

        total_count += 1
        
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    if (total_count-over_len)!=0:
        print("true_acc:",correct_count / (total_count-over_len))

    return {
        "Acc": accuracy
    }
