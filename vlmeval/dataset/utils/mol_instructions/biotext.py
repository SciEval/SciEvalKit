# molecule task
# https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation/molecule

import json
import os
import re
from os import environ
from typing import List, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support

from bert_score import score
from rouge import Rouge



def CER_calculate_f1_score(true_entities, predicted_entities):
    true_entities = set(true_entities.split(', '))
    predicted_entities = set(predicted_entities.split(', '))
    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if len(predicted_entities) > 0 else 0
    recall = true_positive / len(true_entities) if len(true_entities) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # print(true_entities,predicted_entities,f1_score)
    return f1_score

def calculate_f1_score(true_entities, predicted_entities):
    pattern = r'\(.*?\)'
    true_entities = re.findall(pattern, true_entities)
    predicted_entities_tmp = re.findall(pattern, predicted_entities)
    if not predicted_entities_tmp:
        # add () to predicted_entities if it is empty
        predicted_entities = f'({predicted_entities})'
        predicted_entities_tmp = re.findall(pattern, predicted_entities)

    predicted_entities = [entity.strip() for entity in predicted_entities_tmp]

    true_entities = set(true_entities)
    predicted_entities = set(predicted_entities)
    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if len(predicted_entities) > 0 else 0
    recall = true_positive / len(true_entities) if len(true_entities) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def calculate_accuracy_(predictions, references):
    correct_count = 0
    total_count = len(references)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred = pred[0].lower()
        ref = ref[0].lower()
        f1_score = calculate_f1_score(ref, pred)
        correct_count += f1_score

    return correct_count/ total_count

def CER_calculate_accuracy_(predictions, references):
    correct_count = 0
    total_count = len(references)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred = pred[0].lower()
        ref = ref[0].lower()
        f1_score = CER_calculate_f1_score(ref, pred)
        # print(f1_score)
        correct_count += f1_score

    return correct_count/ total_count


def ture_or_false_calculate_accuracy_(predictions, references):
    x,y,z=0,0,0
    correct_count = 0
    total_count = len(references)
    other_answers = 0
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred = pred[0].lower()
        ref = ref[0].lower()
        correct_first_word = ref.split(',')[0].strip().lower()
        # my_first_word = pred.split(',')[0].strip().lower()
        pred = pred.strip().lower()
        if 'yes' in pred:
            my_first_word = 'yes'
        elif 'no' in pred:
            my_first_word = 'no'
        elif 'maybe' in pred or 'may be' in pred or 'might' in pred:
            my_first_word = 'maybe'
        else:
            other_answers += 1
            my_first_word = 'other'
            print(f"Other answer: {pred}, reference: {ref}")

        if correct_first_word=="no" and my_first_word=="no":
            x+=1
        if correct_first_word=="no":
            y+=1
        if my_first_word=="no":
            z+=1
        if correct_first_word == my_first_word:
            correct_count += 1
    accuracy = (correct_count / total_count) * 100
    return accuracy, other_answers


def calculate_macro_f1_(predictions, references):
    correct_answers = [ref[0].split(',')[0].strip().lower() for ref in references]
    my_answers = [pred[0].split(',')[0].strip().lower() for pred in predictions]
    # Compute precision, recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(correct_answers, my_answers, labels=["yes", "no", "maybe"], average=None)
    # Calculate macro F1 by averaging F1-scores for all classes
    macro_f1 = sum(f1) / len(f1)

    return macro_f1

def multi_choice_question_calculate_accuracy(question_data):
    correct_count = 0
    total_count = len(question_data)
    for i,question in enumerate(question_data):
        correct_answer = question["output"].split("(")[1].split(")")[0]
        my_answer=question["my_output"][0]
        if '(A' in question["my_output"] or 'A)' in question["my_output"] or ' A ' in question["my_output"]:
            my_answer = 'A'
        elif '(B' in question["my_output"] or 'B)' in question["my_output"] or ' B ' in question["my_output"]:
            my_answer = 'B'
        elif '(C' in question["my_output"] or 'C)' in question["my_output"] or ' C ' in question["my_output"]:
            my_answer = 'C'
        elif '(D' in question["my_output"] or 'D)' in question["my_output"] or ' D ' in question["my_output"]:
            my_answer = 'D'
        if correct_answer == my_answer:
                correct_count += 1
    accuracy = (correct_count / total_count) * 100

    return accuracy

def multi_choice_question_calculate_accuracy_(predictions, references):
    correct_count = 0
    total_count = len(references)
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        correct_answer = ref[0].split("(")[1].split(")")[0]
        my_answer=pred[0]
        if '(A' in pred[0] or 'A)' in pred[0] or ' A ' in pred[0]:
            my_answer = 'A'
        elif '(B' in pred[0] or 'B)' in pred[0] or ' B ' in pred[0]:
            my_answer = 'B'
        elif '(C' in pred[0] or 'C)' in pred[0] or ' C ' in pred[0]:
            my_answer = 'C'
        elif '(D' in pred[0] or 'D)' in pred[0] or ' D ' in pred[0]:
            my_answer = 'D'
        if correct_answer == my_answer:
                correct_count += 1
    accuracy = (correct_count / total_count) * 100

    return accuracy






