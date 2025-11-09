import warnings
import pandas as pd
import re
from abc import abstractmethod
from sklearn.metrics import matthews_corrcoef, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, mean_absolute_error
from ..smp import *
from .text_base import TextBaseDataset
from .utils.bioinstruction import *



class Bioinstruction(TextBaseDataset):
    TYPE = 'TEXT'
    DATASET_URL = {
        'sirnaEfficiency': '/scievalkit_benchmark/Bioinstruction/sirnaEfficiency.tsv'',
        'Thermostability':'/scievalkit_benchmark/Bioinstruction/Thermostability.tsv'',
        'Fluorescence':'/scievalkit_benchmark/Bioinstruction/Fluorescence.tsv'',
        'Isoform':'/scievalkit_benchmark/Bioinstruction/Isoform.tsv'',
        'MeanRibosomeLoading':'/scievalkit_benchmark/Bioinstruction/MeanRibosomeLoading.tsv'',
        'CRISPROnTarget':'/scievalkit_benchmark/Bioinstruction/CRISPROnTarget.tsv'',
        'Stability':'/scievalkit_benchmark/Bioinstruction/Stability.tsv'',

        'promoter_enhancer_interaction':'/scievalkit_benchmark/Bioinstruction/promoter_enhancer_interaction.tsv'',
        'rna_protein_interaction':'/scievalkit_benchmark/Bioinstruction/rna_protein_interaction.tsv'',
        'emp':'/scievalkit_benchmark/Bioinstruction/emp.tsv'',
        'Solubility':"/scievalkit_benchmark/Bioinstruction/Solubility.tsv'",
        'tf_m':'/scievalkit_benchmark/Bioinstruction/tf_m.tsv'',
        'antibody_antigen':'/scievalkit_benchmark/Bioinstruction/antibody_antigen.tsv',

        'NoncodingRNAFamily':'/scievalkit_benchmark/Bioinstruction/NoncodingRNAFamily.tsv'',
        'ProgrammableRNASwitches':'/scievalkit_benchmark/Bioinstruction/ProgrammableRNASwitches.tsv'',
        'Modification':'/scievalkit_benchmark/Bioinstruction/Modification.tsv'',
        'FunctionEC':'/scievalkit_benchmark/Bioinstruction/FunctionEC.tsv'',
        'enhancer_activity':'/scievalkit_benchmark/Bioinstruction/enhancer_activity.tsv''
    }
    DATASET_MD5 = {
        'sirnaEfficiency': '',
        'Thermostability': '',
        'Fluorescence': '',
        'Isoform': '',
        'MeanRibosomeLoading': '',
        'CRISPROnTarget': '',
        'Stability': '',
        'promoter_enhancer_interaction': '','rna_protein_interaction': '','emp': '','Solubility': '','tf_m': '','antibody_antigen': '',
        'NoncodingRNAFamily': '','ProgrammableRNASwitches': '','Modification': '','FunctionEC': '','enhancer_activity': ''


    }
    regression_spearman=['Stability','Fluorescence','Thermostability','CRISPROnTarget']
    regression_r2=['MeanRibosomeLoading','Isoform']
    regression_task= regression_spearman+regression_r2+['sirnaEfficiency']
    binary_classification_acc=['Solubility']
    binary_classification_mcc=['antibody_antigen','rna_protein_interaction','emp','tf_m','promoter_enhancer_interaction']
    binary_classification_task=binary_classification_mcc+binary_classification_acc
    multi_task=['ProgrammableRNASwitches','Modification','FunctionEC','enhancer_activity']


    def process_regression_task(task_entries):
        result_values = []
        label_values = []
        task_processed_data = []
        num_all=len(task_entries)
        for index,entry in task_entries.iterrows():
            extracted_result = extract_numeric_values(entry["prediction"])
            label = float(entry["category"])
            if len(extracted_result) == 0:
                result_values.append(np.inf)
            else:
                result_values.append(extracted_result[-1])
            label_values.append(label)
        return label_values, result_values

    def process_binary_classification_task(task_entries):
        label_classes = []
        result_classes = []
        task_processed_data = []
        entries_for_model = []
        num_all=len(task_entries)
        for index,entry in task_entries.iterrows():
            label_class = 1 if entry["category"] == 'positive' else 0
            model_output = str(entry["prediction"])
            result_class = None
            score = 0
            if model_output is None:
                result_class = 1 - label_class
            else:
                keyword_result = classify_by_keywords(model_output)
                if keyword_result == "dont_know":
                    result_class = 1 - label_class
                elif keyword_result is not None:
                    result_class = keyword_result
                else:
                    if model_output and model_output.strip():
                        entries_for_model.append({"index": index, "text": model_output})
                    else:
                        result_class = 1 - label_class
            task_processed_data.append({
            "input": entry["question"],
            "original_label": entry["category"],
            "processed_label": label_class,
            "original_model_output": model_output,
            "processed_model_output": result_class, 
            "score": "N/A"
        })
        if entries_for_model:
            texts_to_classify = [item['text'] for item in entries_for_model]
            model_results = classify_by_sentiment_model(texts_to_classify)
            for i, model_item in enumerate(tqdm(entries_for_model)):
                original_index = model_item['index']
                result_class, score = model_results[i]
                task_processed_data[original_index]["processed_model_output"] = result_class
                task_processed_data[original_index]["score"] = str(score)
        result_classes = [d["processed_model_output"] for d in task_processed_data]
        label_classes = [d["processed_label"] for d in task_processed_data]

        return label_classes, result_classes

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        with open("/scievalkit_benchmark/Bioinstruction/ec_labels.json", "r") as f:
            ec_labels = json.load(f)
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
        print(dataset_name)
        print(data['prediction'][3])
        if dataset_name in self.regression_task:
            label_values, result_values=self.process_regression_task(data)
            if dataset_name in self.regression_spearman:
                metrics= compute_spearman(label_values, result_values)
            elif dataset_name in self.regression_r2:
                metrics = compute_R2(label_values, result_values)
            else:
                metrics=compute_mixed_score(label_values,result_values)
            return metrics
        elif dataset_name in self.binary_classification_task:
            label_values, result_values=self.process_binary_classification_task(data)
            if dataset_name in self.binary_classification_mcc:
                metrics=compute_MCC(label_values,result_values)
            else:
                metrics=compute_Acc(label_values,result_values)
            return metrics
        elif dataset_name in self.multi_task:
            if dataset_name=='ProgrammableRNASwitches':
                metrics= compute_R2_for_ProgrammableRNASwitches_task(data)
            elif dataset_name=='enhancer_activity':
                metrics= compute_PCC_for_enhancer_activity_task(data)
            elif dataset_name=='FunctionEC':
                metrics= compute_Fmax_for_FunctionEC_task(data,ec_labels)
            else:
                metrics= compute_AUC_for_Modification_task(data)
            return metrics
        elif dataset_name=='NoncodingRNAFamily':
            metrics= compute_Acc_for_NoncodingRNAFamily_task(data)
            return metrics
        else:
            print('Error Task!')
            pass
        return
