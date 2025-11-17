# molecule task
# https://github.com/zjunlp/Mol-Instructions/tree/main/evaluation/molecule

import json
import os
import re
from os import environ
import numpy as np
from sklearn.metrics import mean_absolute_error
from datasets import Dataset, DatasetDict


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import selfies as sf
from Levenshtein import distance as lev
import glob


def convert_to_canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None


def compute_MAE_property_prediction_str(predictions, references):
    y_pred = np.array([float(p[0]) for p in predictions])
    y_true = np.array([float(r[0]) for r in references])
    mae = mean_absolute_error(y_true, y_pred) * 1000 # scale to match the presentation of Opencompass
    return {'mae': mae}

def compute_fingerprint_metricts(predictions, references, morgan_r=2,):
    bad_mols = 0
    outputs = []

    for pred, refer in zip(predictions, references):
        try:
            if pred[0] is None:
                bad_mols += 1
                continue
            pred_ = Chem.MolFromSmiles(pred[0])
            refer_ = Chem.MolFromSmiles(refer[0])
            if pred_ is None:
                # print(pred)
                bad_mols += 1
                continue
            outputs.append((refer_, pred_))
        except Exception as e:
            import pdb; pdb.set_trace()

    validity_score = len(outputs) / (len(outputs) + bad_mols)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (gt_m, ot_m) in enumerate(enum_list):
        # if i % 100 == 0:
        #     if verbose: print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m),
                                                            metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m),
                                                          metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m, morgan_r),
                                                          AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)

    return {
        'validity_score': validity_score,
        'maccs_sims_score': maccs_sims_score,
        'rdk_sims_score': rdk_sims_score,
        'morgan_sims_score': morgan_sims_score
    }

def compute_mol_translation_selfies(predictions, references):
    outputs = []
    bad_mols = 0
    for pred, refer in zip(predictions, references):
        if pred[0] is None:
            bad_mols += 1
            continue
        pred_canonical_smiles = pred[0]
        refer_canonical_smiles = refer[0]
        try:
            pred_sf = sf.encoder(pred_canonical_smiles)
            refer_sf = sf.encoder(refer_canonical_smiles)
        except:
            bad_mols += 1
            continue

        outputs.append((refer_sf, pred_sf, refer_canonical_smiles, pred_canonical_smiles))

    bleu_self_scores = []
    bleu_smi_scores = []

    references_self = []
    hypotheses_self = []

    references_smi = []
    hypotheses_smi = []

    for i, (gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        # if i % 100 == 0:
        #     if verbose:
        #         print(i, 'processed.')

        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self]

        references_self.append([gt_self_tokens])
        hypotheses_self.append(out_self_tokens)

        gt_smi_tokens = [c for c in gt_smi]
        ot_smi_tokens = [c for c in ot_smi]

        references_smi.append([gt_smi_tokens])
        hypotheses_smi.append(ot_smi_tokens)

    # BLEU score
    bleu_score_self = corpus_bleu(references_self, hypotheses_self)

    references_self = []
    hypotheses_self = []

    references_smi = []
    hypotheses_smi = []

    levs_self = []
    levs_smi = []

    num_exact = 0

    # bad_mols = 0

    for i, (gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)

        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
            # if gt == out: num_exact += 1 #old version that didn't standardize strings
        except:
            bad_mols += 1

        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))

    # Exact matching score
    exact_match_score = num_exact / (i + 1)
    # if verbose:
    #     print('Exact Match:')
    #     print(exact_match_score)

    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    # if verbose:
    #     print('SMILES Levenshtein:')
    #     print(levenshtein_score_smi)

    return {
        'bleu_self_scores': bleu_score_self,
        # 'bleu_smi_scores': corpus_bleu(references_smi, hypotheses_smi),
        'exact_match_score': exact_match_score,
        'levenshtein_score_smi': levenshtein_score_smi,
    }

    """修复SMILES字符串中缺失的右括号"""
    if not isinstance(smiles, str):
        return smiles

    # 计算左括号和右括号的数量差
    left_count = smiles.count('(')
    right_count = smiles.count(')')
    missing = left_count - right_count

    if missing > 0:
        return smiles + ')' * missing
    return smiles

# class Mol_Instructions_Evaluator_Mol(BaseEvaluator):
    def __init__(self, task, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        if not isinstance(predictions[0], list):
            predictions = [[pred] for pred in predictions]
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]
        # import pdb;pdb.set_trace()
        task = self.task
        pred_list = predictions
        gold_list = references

        if task in ('property_prediction_str',):
            results = compute_MAE_property_prediction_str(pred_list, gold_list)
        elif task in ('description_guided_molecule_design', 'forward_reaction_prediction', 'retrosynthesis',
                      'reagent_prediction'):
            fingerprint_metrics = compute_fingerprint_metricts(pred_list, gold_list)
            mol_translation_selfies = compute_mol_translation_selfies(pred_list, gold_list)
            # Combine the results from both computations
            results = {**fingerprint_metrics, **mol_translation_selfies}
            # change the order to 'exact', 'blue', 'levenshtein', 'RDK', 'MACCS', 'Morgan', 'validity'
            results = {
                'exact_match_score': results['exact_match_score'],
                'bleu_self_scores': results['bleu_self_scores'],
                'levenshtein_score_smi': results['levenshtein_score_smi'],
                'rdk_sims_score': results['rdk_sims_score'],
                'maccs_sims_score': results['maccs_sims_score'],
                'morgan_sims_score': results['morgan_sims_score'],
                'validity_score': results['validity_score']
            }
        elif task in ('molecular_description_generation',):
            results = compute_text_translation_metrics(pred_list, gold_list)
        else:
            raise ValueError(task)

        return results

def compute_text_translation_metrics(predictions, references, text_model='allenai/scibert_scivocab_uncased', text_trunc_length=512):
    outputs = []

    for pred, refer in zip(predictions, references):
        try:
            pred_ = pred[0].rsplit('.', 1)[0] + '.' if isinstance(pred[0], str) else pred[0]
            outputs.append((refer[0], pred_))
        except:
            import pdb;pdb.set_trace()

    text_tokenizer = BertTokenizerFast.from_pretrained('/mnt/shared-storage-user/ai4sreason/shared/huggingface/cache/huggingface/hub/models--allenai--scibert_scivocab_uncased/snapshots/24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1/')

    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        # if i % 100 == 0: print(i, 'processed.')

        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    # print('BLEU-2 score:', bleu2)
    # print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    # print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    # print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
    # print('rouge1:', rouge_1)
    # print('rouge2:', rouge_2)
    # print('rougeL:', rouge_l)
    return {
        'bleu2': bleu2,
        'bleu4': bleu4,
        'meteor_score': _meteor_score,
        'rouge1': rouge_1,
        'rouge2': rouge_2,
        'rougeL': rouge_l
    }

