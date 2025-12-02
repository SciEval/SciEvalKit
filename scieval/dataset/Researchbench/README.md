# Researchbench

本项目支持三类主要子任务：

* **Retrieve 任务**：从候选文献中筛选灵感论文。
* **Generate 任务**：基于科研背景生成新的研究想法或摘要。
* **Rank 任务**：对候选研究结果进行排序或偏好评估。

本文档介绍环境配置、命令行用法、配置文件示例以及典型输出格式。

---

## 一、环境设置

在运行任务前，请先设置以下环境变量：

```bash
export BASE_URL="你的_api_base"
export MODEL="你的模型名称"
export API_KEY="你的_api_key"
```

如果项目目录中存在 `.env` 文件，也会自动读取其中的配置。

---

## 二、Retrieve 任务

### 1. 运行命令示例

```bash
python run.py \
  --config configs/rb_run_retrieve.json \
  --work-dir outputs/researchbench_retrieve \
  --verbose --reuse
```

### 2. 配置文件示例

```json
{
  "model": {
    "GeminiFlash_viaOpenAICompat": {
      "class": "GPT4V",
      "model": "gemini-2.5-flash",
      "api_base": "<your_api_base>"
    }
  },
  "data": {
    "ResearchbenchRetrieve": {
      "class": "ResearchbenchRetrieve",
      "dataset": "ResearchbenchRetrieve",
      "tsv_path": "datasets/researchbench_retrieve.tsv"
    }
  }
}
```

### 3. 输出结果示例

```text
100%|████████████████████████████████████████████████████| 23/23 [00:14<00:00, 1.58it/s]
ResearchbenchRetrieve_tiny23

Evaluation Results:
{
    "dataset": "researchbench_retrieve",
    "size": 23,
    "weighted": false,
    "hit@1": 0.087,
    "hit@3": 0.174
}
```

### 4. 说明

* `hit@1` 与 `hit@3` 表示前 1 篇 / 前 3 篇命中率。
* `--reuse` 参数表示复用上一次预测结果（如缓存的 Excel 文件）。
* 输出文件位于 `outputs/researchbench_retrieve/...` 目录下。

---

## 三、Generate 任务

### 1. 运行命令示例

```bash
python run.py \
  --config configs/rb_run_generate.json \
  --mode all \
  --verbose \
  --work-dir outputs/researchbench_generate
```

### 2. 配置文件示例

```json
{
  "model": {
    "GeminiFlash_viaOpenAICompat": {
      "class": "GPT4V",
      "model": "gemini-2.5-flash",
      "api_base": "<your_api_base>"
    }
  },
  "data": {
    "ResearchbenchGenerate": {
      "class": "ResearchbenchGenerate",
      "ann_path": "datasets/researchbench_generate.tsv",
      "save_dir": "outputs/researchbench_generate"
    }
  }
}
```

### 3. 输出结果示例

```text
100%|████████████████████████████████████████████████████| 1084/1084 [00:11<00:00, 1.15it/s]
ResearchbenchGenerate

[OK] LLM Judge Finished, AVG Score: 2.615
Save results at: outputs/researchbench_generate/GeminiFlash_viaOpenAICompat/.../judged.xlsx

Evaluation Results:
{
    "items_scored": 1084,
    "avg_score": 2.615,
    "judged_file": "outputs/researchbench_generate/.../judged.xlsx",
    "judge_model": "gpt-4o-mini"
}
```

### 4. 说明

* `avg_score` 表示生成任务的平均得分，范围通常为 0–5。
* `score_dist` 给出了不同分值的分布情况。
* `judged_file` 为评测结果文件路径，包含每个样本的详细得分。

---

## 四、Rank 任务

### 1. 运行命令示例

```bash
python run.py \
  --config configs/rb_run_rank.json \
  --mode all \
  --verbose
```

### 2. 配置文件示例

```json
{
  "model": {
    "GeminiFlash_viaOpenAICompat": {
      "class": "GPT4V",
      "model": "gemini-2.5-flash",
      "api_base": "<your_api_base>"
    }
  },
  "data": {
    "ResearchbenchRank": {
      "class": "ResearchbenchRank",
      "dataset": "ResearchbenchRank",
      "ann_path": "datasets/rank.tsv",
      "save_dir": "outputs/researchbench_rank"
    }
  }
}
```

### 3. 输出结果示例

```text
100%|███████████████████████████████████████████████████████████████████| 195/195 [01:02<00:00, 3.12it/s]
ResearchbenchRank

Evaluation Results:
{
    "overall": {
        "num_pairs": 195,
        "num_parsable": 195,
        "pairwise_acc": 0.728,
        "mean_rank_position": 5.077,
        "mean_rank16": 11.923,
        "mean_rank_score": 0.683
    }
}
```

### 4. 说明

* `pairwise_acc`：两两比较准确率，用于衡量模型排序的一致性。
* `mean_rank_position`：平均排名位置。
* `mean_rank_score`：综合排名得分。
* 输出结果保存于 `outputs/researchbench_rank/...` 目录中。
