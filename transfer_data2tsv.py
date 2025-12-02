import pandas as pd
from datasets import load_dataset
import json


def export_scicode_to_tsv(output_file="C:\\Users\\86198\\LMUDatascicode_flattened.tsv"):
    print("正在下载并处理数据...")
    # 1. 加载原始数据
    try:
        hf = load_dataset("SciCode1/SciCode", split="test")  # 或者是 "validation"
    except Exception as e:
        print(f"下载失败: {e}")
        return

    rows = []
    idx = 0
    # 2. 执行原本的 Flatten 逻辑
    for prob in hf:
        pid = prob["problem_id"]
        subs = prob["sub_steps"]
        total = len(subs)
        for s_idx, _ in enumerate(subs):
            rows.append({
                "index": idx,
                "id": f"{pid}.{s_idx + 1}",
                "problem_id": pid,
                "step": s_idx + 1,
                "tot_steps": total,
                # 【关键】这里不直接存对象，而是存 JSON 字符串，防止格式错乱
                "record": json.dumps(prob, ensure_ascii=False)
            })
            idx += 1

    # 3. 转为 DataFrame
    df = pd.DataFrame(rows)

    # 4. 保存为 TSV
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"成功导出 TSV 文件: {output_file}，共 {len(df)} 行。")


# 执行导出
if __name__ == "__main__":
    export_scicode_to_tsv()