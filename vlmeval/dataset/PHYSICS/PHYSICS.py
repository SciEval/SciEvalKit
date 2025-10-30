import numpy as np
from jinja2 import Template
import os
import json
from ..text_base import TextBaseDataset
from vlmeval.smp.file import load
from tqdm import tqdm
from .reward_score  import compute_score
import signal
from contextlib import contextmanager
from .reward_score import Model_args


def generate_score_report(jsonl_file_path: str):
    if not os.path.isfile(jsonl_file_path):
        print(f"错误: 文件未找到 -> {jsonl_file_path}")
        return

    input_directory = os.path.dirname(jsonl_file_path)
    input_filename = os.path.basename(jsonl_file_path)

    report_path = os.path.join(input_directory, "score_report.txt")

    total_records = 0
    acc_true_count = 0
    rule_based_acc_true_count = 0

    print(f"正在处理文件: {input_filename}...")

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    total_records += 1

                    if data.get('acc') is True:
                        acc_true_count += 1

                    if data.get('rule_based_acc') is True:
                        rule_based_acc_true_count += 1

                except json.JSONDecodeError:
                    print(f"警告: 发现无效的JSON行，已跳过: {line.strip()}")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    if total_records > 0:
        acc_percentage = (acc_true_count / total_records) * 100
        rule_based_acc_percentage = (rule_based_acc_true_count / total_records) * 100
    else:
        acc_percentage = 0
        rule_based_acc_percentage = 0
        print("警告: 文件为空或不包含任何有效的JSON记录。")

    report_content = (
        f"Score Report for: {input_filename}\n"
        f"========================================\n"
        f"Total Records Processed: {total_records}\n\n"
        f"Accuracy ('acc'):\n"
        f"  - True Count: {acc_true_count}\n"
        f"  - Correctness Rate: {acc_percentage:.2f}%\n\n"
        f"Rule-Based Accuracy ('rule_based_acc'):\n"
        f"  - True Count: {rule_based_acc_true_count}\n"
        f"  - Correctness Rate: {rule_based_acc_percentage:.2f}%\n"
    )

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print("-" * 40)
        print(f"报告生成成功！已保存至: {report_path}")
    except Exception as e:
        print(f"写入报告文件时发生错误: {e}")

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        
class PHYSICS(TextBaseDataset):
    TYPE = "QA"
    DATASET_URL = {
        'PHYSICS-test': 'https://huggingface.co/datasets/desimfj/PHYSICS.tsv'
    }

    DATASET_MD5 = {
        'PHYSICS-test': '7303c8d8bcb11f78f420aa25216cc9ae'
    }

    # def load_data(path, read_num=None, repeat_time=1):
    #     name = os.path.basename(path).replace('.jsonl', '')
    #     df = pd.read_json(path, lines=True)
    #     df['dataset'] = name
    #
    #     # 采样（若指定 read_num）
    #     sample_kwargs = {'n': read_num} if read_num else {'frac': 1}
    #     df = df.sample(**sample_kwargs).drop_duplicates(subset=['question']).reset_index(drop=True)
    #
    #     # 重复（若指定 repeat_time）
    #     if repeat_time > 1:
    #         df = df.loc[np.repeat(df.index, repeat_time)].reset_index(drop=True)
    #
    #     return df


    def build_prompt(self, line):
        SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it.
    The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags."""
        
        INSTRUCTION_TEMPLATE = """Below is an open-ended problem in undergraduate-level Physics. Please answer this problem adhering to the following rules:
    1. Use LaTeX format for formulas.
    2. Put the final answer(s) in \\boxed{}, without units.
    3. If multiple answers exist, separate them by commas in \\boxed{}.
    Problem: {{prompt}}"""

        # 获取问题文本（支持 Series 或 dict）
        prompt_text = line["question"] if isinstance(line, dict) else line

        # 应用指令模板
        prompt_text = Template(INSTRUCTION_TEMPLATE).render(prompt=prompt_text)

        # 拼接最终的完整 prompt
        full_prompt = (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User: {prompt_text}\n"
        )

        return full_prompt
    
    def write_jsonl(self, data_path, dataset, indent=0, mode='w'):
        with open(data_path, mode, encoding='UTF-8') as f:
            if not isinstance(dataset, list):
                dataset = [dataset]
            for data in dataset:
                line = json.dumps(data, ensure_ascii=False, indent=indent if indent != 0 else None)
                f.write(line + '\n')
            
    def evaluate(self, eval_file, **judge_kwargs):
        directory = os.path.dirname(eval_file)
        basename = os.path.basename(eval_file)
        file_stem, _ = os.path.splitext(basename)
        output_filename = f"{file_stem}_results.jsonl"
        output_path = os.path.join(directory, output_filename)
        os.makedirs(directory, exist_ok=True)

        model_args = Model_args()
        model_args.base_url = judge_kwargs['model_args']['base_url']
        model_args.api_key = judge_kwargs['model_args']['api_key']
        model_args.model_name = judge_kwargs['model_args']['model_name']
        model_args.temperature = judge_kwargs['model_args']['temperature']

        data = load(eval_file)
        for item in tqdm(data.iterrows(), desc="Scoring"):
            # result = compute_score(item['model_output'], item['ground_truth'], item['problem']) # olympiadbench
            try:
                with timeout(30):
                    item = item[1].to_dict()

                    result = compute_score(item['prediction'], item['answer'], item['question'], model_args = model_args)
                    item['rule_based_acc'] = result['rule_based_acc']
                    item['acc'] = result['acc']
                    item['extracted_gt'] = result['extracted_gt']
                    item['extracted_pred'] = result['extracted_pred']
                    self.write_jsonl(output_path, item, mode='a')
            except TimeoutError:
                print(f"Timeout processing item: {item.get('question', 'Unknown')}")
                continue
        generate_score_report(output_path)
        return eval_file