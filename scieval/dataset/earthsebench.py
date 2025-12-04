import string
from scieval import *
from ..smp import *
from ..smp.file import get_intermediate_file_path
from .text_base import TextBaseDataset
from .utils.judge_util import *
from ..utils import track_progress_rich

EVAL_TEMPLATE = """
Please act as an expert evaluator and determine which of the following two answers is better.

**Evaluation Criteria:**
1. Assess how well each answer addresses the original question. Closer alignment is better.
2. Evaluate the scientific accuracy and logical coherence of each answer. More rigorous and professional reasoning is preferred.
3. Consider the relevance and depth of detail. More relevant and well-supported details indicate a better answer.
4. It is not the case that the longer the answer, the better. If the answer is long but does not meet the above requirements, it is not a good answer.

**Instructions:**
1. Do **not** generate a new answer to the original question. Your task is only to evaluate the two provided answers.
2. Based on the criteria above, choose which answer is better.
3. Your response must be **only** one letter: `A` or `B`.
4. Do **not** provide explanations, commentary, or corrections, even if there are errors in the inputs.
5. This is purely an evaluation task.

**[Question Start]**
{question}
**[Question End]**

**[Answer A Start]**
{answer}
**[Answer A Start]**

**[Answer B Start]**
{prediction}
**[Answer B Start]**

**The better answer is:**
"""


def report_score(df):
    # assert group in [None, 'category']
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'category']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['score']) * 100 for sp in res['split']]
        elif group not in df:
            continue
        else:
            abilities = list(set(df[group]))
            abilities.sort()
            for ab in abilities:
                sub_df = df[df[group] == ab]
                res[ab] = [np.mean(sub_df[sub_df['split'] == sp]['score']) * 100 for sp in res['split']]
    return pd.DataFrame(res)


def make_prompt(line):
    question = line['question']
    answer = line['answer']
    tmpl = EVAL_TEMPLATE
    prompt = tmpl.format(
        question=question,
        answer=answer,
        prediction=line['prediction']
    )
    return prompt


def EarthSE_auxeval(model, data):
    if isinstance(data, pd.DataFrame) and len(data) > 1:
        lt = len(data)
        for i in range(lt):
            total_score = 0
            item = data.iloc[i]
            if item['answer'] == item ['prediction']:
                total_score += 1
                continue
            prompt = make_prompt(item)
            retry = 3
            for j in range(retry):
                output = model.generate(prompt, temperature=0.5 * j)
                if output in ['A', 'B']:
                    if output == 'A':
                        total_score += 0
                    elif output == 'B':
                        total_score += 1
                    break
        avg_score = total_score / lt
        return dict(score=avg_score, log='Success to Judge')
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        prompt = make_prompt(item)
        retry = 3
        for i in range(retry):
            if item['answer'] == item ['prediction']:
                score = 1
                return dict(score=score, log='Success to Judge')
            output = model.generate(prompt, temperature=0.5 * i)
            if output in ['A', 'B']:
                if output == 'A':
                    score = 0
                elif output == 'B':
                    score = 1
                return dict(score=score, log='Success to Judge')
        return dict(score=0, log='Fail to Judge')


class EarthSE(TextBaseDataset):
    TYPE = 'QA'

    DATASET_URL = {
        'EarthSE': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/EarthSE.tsv', # TODO upload data
    }

    DATASET_MD5 = {
        'EarthSE': '896dc4896f01e4d89fbb076c38677f03',
    }

    FREE_FORM_PROMPT = '''
Please answer the following question:

{ques}

The answer is:
'''

    MULTIPLE_CHOICE_PROMPT = '''
Please respond to the following multiple-choice question by providing your answer as a single letter, without any additional text.

{ques}

The answer is (single letter):
'''

    FILL_IN_THE_BLANK_PROMPT = '''
Please answer the fill-in-the-blank question below with lowercase words or phrases. If your answer contains multiple words or phrases, please separate them with commas. No additional text is required.

{ques}

The answer is:
'''

    TRUE_FALSE_PROMPT = '''
Please answer the following true or false question with "True" or "False" without adding any additional text.

{ques}

The answer is ("True" or "False"):
'''

    
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question_type = line['question_type']
        question = line['question']

        if question_type == 'free_form':
            question = self.FREE_FORM_PROMPT.format(ques=question)
        elif question_type == 'multiple_choice':
            question = self.MULTIPLE_CHOICE_PROMPT.format(ques=question)
        elif question_type == 'fill_in_the_blank':
            question = self.FILL_IN_THE_BLANK_PROMPT.format(ques=question)
        elif question_type == 'true_false':
            question = self.TRUE_FALSE_PROMPT.format(ques=question)

        msgs = [{'type': 'text', 'value': question}]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        storage = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.pop('model', 'GPT4o_20241120')
            if model == 'exact_matching':
                model = None
            else:
                try:
                    model = build_judge_model(model=model, **judge_kwargs)
                    if not model.working():
                        warnings.warn(
                            f'Judge model {model} is not working properly (working() returned False), will use exact matching for evaluation.')
                        model = None

                except Exception as e:
                    warnings.warn(
                        f'Failed to initialize judge model {model}: {e}. Will use exact matching for evaluation.')
                    model = None

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    res = track_progress_rich(
                        EarthSE_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['score'] = [x['score'] for x in judge_results]
            dump(data, storage)
        data = load(storage)
        score = report_score(data)

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score, score_file)
        return score
