import numpy as np
import pandas as pd
from collections import defaultdict
from ...smp import *

from .SEED import SEED

def extract_boxed_latex(text):
    if not isinstance(text, str):
        return ""
    start = text.find(r'\boxed{')
    if start == -1:
        return ""
    start += len(r'\boxed{')

    depth = 1
    end = start
    while end < len(text):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
        end += 1
    return ""

def report_cmphys_score(df):
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
    df['hit'] = pd.to_numeric(df['hit'], errors='coerce').fillna(0)
    
    df['score_norm'] = df['score'] / 100.0
    
    report = defaultdict(lambda: {'Avg. Score': 0.0, 'Hit Rate': 0.0, 'Count': 0})
    
    groups = {
        'Overall': df,
    }
    
    if 'topic' in df:
        for topic in sorted(df['topic'].unique()):
            groups[topic] = df[df['topic'] == topic]

    if 'answer_type' in df:
        for atype in sorted(df['answer_type'].unique()):
            groups[f"Type_{atype}"] = df[df['answer_type'] == atype]

    for name, group_df in groups.items():
        if not group_df.empty:
            report[name]['Avg. Score'] = np.mean(group_df['score_norm']) * 100  
            report[name]['Hit Rate'] = np.mean(group_df['hit']) * 100         
            report[name]['Count'] = len(group_df)
    
    report_df = pd.DataFrame.from_dict(report, orient='index')
    
    return report_df

class CMPhysBench():
    """
    CMPhysBench Dataset.
    This dataset requires special evaluation logic (SEED metric).
    """
    @classmethod
    def supported_datasets(cls):
        return {
            'weidawang/CMPhysBench': cls,
        }
        
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question']
        message =[
                {
                    "role": "system",
                    "content": "You are a condensed matter physics expert. Please read the following question and provide a step-by-step solution using only the given symbols. Do not introduce any new symbols that are not provided in the problem statement. Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{} environment."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        return message


    def evaluate(self, eval_file, **kwargs):

        results = load(eval_file)
        assert all(col in results for col in ['answer', 'prediction', 'answer_type', 'topic']), \
            "Evaluation file must contain 'answer', 'prediction', 'answer_type', and 'topic' columns."
        
        for col in ['prediction', 'answer', 'answer_type', 'topic']:
            results[col] = [str(x) for x in results[col]]
            
        results['hit'] = 0  
        results['log'] = "" 
        results['score'] = 0.0 
        
        for i in range(len(results)):
            llm_response = results.loc[i, 'prediction']
            prediction = extract_boxed_latex(llm_response)
            
            ground_truth = results.loc[i, 'answer']
            answer_type = results.loc[i, 'answer_type']
            
            if not prediction or pd.isna(prediction):
                log_msg = "Prediction is empty or no boxed answer found."
                results.loc[i, 'log'] = log_msg
                results.loc[i, 'score'] = 0.0
                continue
            
            try:
                eval_result, _, _, _ = SEED(
                    ground_truth=str(ground_truth),
                    prediction=str(prediction),
                    answer_type=str(answer_type)
                )

                score = eval_result.get('score', 0)
                log = eval_result.get('log', '')
                
                results.loc[i, 'score'] = score
                results.loc[i, 'log'] = log
                
                if score == 100:
                    results.loc[i, 'hit'] = 1

            except Exception as e:
                error_msg = f"Error during evaluation: {e}"
                results.loc[i, 'log'] = error_msg
                results.loc[i, 'score'] = 0.0

        dump(results, eval_file)

        score_df = report_cmphys_score(results)

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score_df, score_file)

        return score_df