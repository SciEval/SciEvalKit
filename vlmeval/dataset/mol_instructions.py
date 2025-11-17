import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from ..smp import *
from .text_base import TextBaseDataset
from .utils.mol_instructions.biotext import *    #导入辅助函数
from .utils.mol_instructions.molecule import *
from .utils.mol_instructions.protein import *
import evaluate

class Mol_Instructions(TextBaseDataset):
    TYPE = 'TEXT'
    tsv_root = '/mnt/shared-storage-user/ai4sreason/wangyizhou/Evalkit_tools/tsv_files'
    DATASET_URL = {
        'chemical_disease_interaction_extraction': f'{tsv_root}/chemical_disease_interaction_extraction.tsv',
        'chemical_entity_recognition': f'{tsv_root}/chemical_entity_recognition.tsv',
        'chemical_protein_interaction_extraction': f'{tsv_root}/chemical_protein_interaction_extraction.tsv',
        'multi_choice_question': f'{tsv_root}/multi_choice_question.tsv',
        'open_question': f'{tsv_root}/open_question.tsv',
        'true_or_false_question': f'{tsv_root}/true_or_false_question.tsv',
        'property_prediction_str': f'{tsv_root}/property_prediction_str.tsv',
        'description_guided_molecule_design': f'{tsv_root}/description_guided_molecule_design.tsv',
        'forward_reaction_prediction': f'{tsv_root}/forward_reaction_prediction.tsv',
        'retrosynthesis': f'{tsv_root}/retrosynthesis.tsv',
        'reagent_prediction': f'{tsv_root}/reagent_prediction.tsv',
        'molecular_description_generation': f'{tsv_root}/molecular_description_generation.tsv',
        'catalytic_activity': f'{tsv_root}/catalytic_activity.tsv',
        'domain_motif': f'{tsv_root}/domain_motif.tsv',
        'general_function': f'{tsv_root}/general_function.tsv',
        'protein_function': f'{tsv_root}/protein_function.tsv',
        'protein_design': f'{tsv_root}/protein_design.tsv',
    }
    DATASET_MD5 = {
        'chemical_disease_interaction_extraction': None,
        'chemical_entity_recognition': None,
        'chemical_protein_interaction_extraction': None,
        'multi_choice_question': None,
        'open_question': None,
        'true_or_false_question': None,
        'property_prediction_str': None,
        'description_guided_molecule_design': None,
        'forward_reaction_prediction': None,
        'retrosynthesis': None,
        'reagent_prediction': None,
        'molecular_description_generation': None,
        'catalytic_activity': None,
        'domain_motif': None,
        'general_function': None,
        'protein_function': None,
        'protein_design': None,
    }#MD5码暂时不需要，先置空


    biotext_tasks = ['chemical_disease_interaction_extraction', 'chemical_entity_recognition', 'chemical_protein_interaction_extraction',
                    'multi_choice_question', 'open_question', 'true_or_false_question']

    molecule_tasks = [
        'property_prediction_str',
        'description_guided_molecule_design',
        'forward_reaction_prediction',
        'retrosynthesis',
        'reagent_prediction',
        'molecular_description_generation',
    ]

    protein_tasks = [
        'catalytic_activity',
        'domain_motif',
        'general_function',
        'protein_function',
        'protein_design',
    ]
    

    def process_biotext_tasks(data, dataset_name):
        result_values = []
        label_values = []
        task_processed_data = []
        num_all = len(data)
        
        def remove_think_tag(text):
            text = text.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
            text = text.strip()
            return text

        for index, entry in data.iterrows():
            
            model_output = str(entry['prediction'])
            label = str(entry['answer'])
            label = remove_think_tag(label)
            model_output = remove_think_tag(model_output)


            result_values.append([model_output])
            label_values.append([label])
        
        predictions = result_values
        references = label_values
        
        return references, predictions
    
    def process_protein_tasks(data, dataset_name):
        result_values = []
        label_values = []
        task_processed_data = []
        num_all = len(data)
        
        def remove_think_tag(text, dataset_name):
            text = text.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
            if dataset_name == 'protein_design':
                pattern = r'<protein>(.*?)</protein>'
                match = re.search(pattern, text)
                if match:
                    text = match.group(1)
                    valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    text = ''.join(filter(lambda x: x in valid_letters, text))
                else:
                    text = ''

            text = text.strip()
            return text

        for index, entry in data.iterrows():
            
            model_output = str(entry['prediction'])
            label = str(entry['answer'])
            label = remove_think_tag(label, dataset_name)
            model_output = remove_think_tag(label, dataset_name)


            result_values.append([model_output])
            label_values.append([label])
        
        predictions = result_values
        references = label_values
        
        return references, predictions

    def process_molecule_tasks(data, dataset_name):
        result_values = []
        label_values = []
        task_processed_data = []
        num_all = len(data)
        
        def remove_think_tag(text, dataset_name='property_prediction_str'):
            text = text.strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
            text = text.strip()
            task = dataset_name
            if task == 'property_prediction_str':
                text = re.sub(r'.*?</think>\s*', '', text, flags=re.DOTALL)
                text = re.sub(r'(?<=\d) +(?=\d)|(?<=\.) +(?=\d)', '', text)
                num_match = re.search(r'[-+]?\d*\.\d+|\d+', text)
                text = num_match.group(0) if num_match else 0
            elif task in ['description_guided_molecule_design', 'forward_reaction_prediction','retrosynthesis',
                        'reagent_prediction',]:
                pattern = r'<SMILES>(.*?)</SMILES>'
                match = re.search(pattern, text)
                if match:
                    smiles = match.group(1).strip()
                    text = convert_to_canonical_smiles(smiles)
                else:
                    text = None
            else:
                pass

            return text

        for index, entry in data.iterrows():
            
            model_output = str(entry['prediction'])
            label = str(entry['answer'])
            model_output = remove_think_tag(model_output, dataset_name)
            label = remove_think_tag(label, dataset_name)
            # debug only
            # model_output = label[:2]

            result_values.append([model_output])
            label_values.append([label])
        
        predictions = result_values
        references = label_values
        
        return references, predictions

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        result_values = []
        label_values = []
        data = load(eval_file)
        data= data[~pd.isna(data["prediction"])]
        assert 'answer' in data and 'prediction' in data 
        #获取dataset_name名
        dataset_name = None

        for name in self.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break
         #根据dataset_name选用对应的eval函数
        if dataset_name in self.biotext_tasks:
            references, predictions=self.process_biotext_tasks(data, dataset_name)
            if dataset_name in ('chemical_disease_interaction_extraction',
                         'chemical_protein_interaction_extraction',):
                results = {
                'f1': calculate_accuracy_(predictions, references),
            }
            elif dataset_name in ('chemical_entity_recognition',):
                results = {
                    'f1': CER_calculate_accuracy_(predictions, references),
                }
            elif dataset_name  == 'true_or_false_question':
                acc, other_answers = ture_or_false_calculate_accuracy_(predictions, references)
                results = {
                    'accuracy': acc,
                    'other_answers': other_answers,
                }
            elif dataset_name  == 'multi_choice_question':
                results = {
                    'accuracy': multi_choice_question_calculate_accuracy_(predictions, references),
                }
            elif dataset_name  == 'open_question':
                correct_answers = [ref[0] for ref in references]
                my_answers = [pred[0] for pred in predictions]
                P, R, F1 = score(my_answers, correct_answers, lang='en', verbose=False,) 
                # model_type='/mnt/shared-storage-user/ai4sreason/shared/huggingface/cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59/',
                # num_layers=17)
                results = {
                    'bert_score': sum(F1).item() / len(F1),
                }
            return results
        elif dataset_name in self.molecule_tasks:
            references, predictions=self.process_molecule_tasks(data, dataset_name)
            pred_list = predictions
            gold_list = references
            task = dataset_name
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
        elif dataset_name in self.protein_tasks:
            references, predictions=self.process_protein_tasks(data, dataset_name)
            if dataset_name == 'protein_design':
                scores = []
                for pred, refer in zip(predictions, references):
                    pred = pred[0].strip()
                    refer = refer[0].strip()
                    if not pred or not refer:
                        scores.append(0.0)
                    else:
                        # Calculate the normalized Smith-Waterman score
                        score = normalized_smith_waterman(pred, refer) * 100 # Convert to percentage
                        scores.append(score)

                averaged_valid_scores = [
                    score for score in scores if score > 0
                ]

                results = {
                    'Max SW score': max(scores),
                    'Min SW score': min(scores),
                    'Average SW score': sum(scores) / len(scores),
                    "valid average SW score": sum(averaged_valid_scores) / len(averaged_valid_scores) if averaged_valid_scores else 0.0,
                }
            else:
                metric = evaluate.load('rouge')
                scores = metric.compute(**self._preprocess(predictions, references))
            return results
        else:
            pass
        return 