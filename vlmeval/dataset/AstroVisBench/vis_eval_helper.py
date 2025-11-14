from google import genai
import base64
import pandas as pd
import os
import csv
import json
import tkinter as tk
from tkinter import ttk
import sys
import html
import pprint
import anthropic
from PIL import Image
import re
from glob import glob
from tqdm import tqdm
import copy
import argparse
from openai import OpenAI

prompts_csv = 'vlmeval/dataset/AstroVisBench/viseval_prompts.csv'
model = "claude-sonnet-4-5-20250929"
client = aclient = OpenAI(
    base_url="https://api.boyuerichdata.opensphereai.com/v1", # 确保这里以 /v1 结尾
    api_key=os.getenv("BOYUE_API_KEY")
)


def safe_parse_json_from_claude(text):
    match = re.search(r'(\{\s*"Rationale"\s*:\s*".*?"\s*,\s*"Errors"\s*:\s*".*?"\s*\})', text, re.DOTALL)
    if not match:
        return "", ":x: No JSON block found"
    json_block = match.group(1)
    try:
        clean_block = re.sub(r'(?<!\\)[\x00-\x1F]', '', json_block)
        parsed = json.loads(clean_block)
        rationale = parsed.get("Rationale", "").replace("\n", " ").strip()
        errors = parsed.get("Errors", "").replace("\n", " ").strip()
        return rationale, errors
    except Exception as e:
        return "", f":x: JSON parse error: {e}"


def query_model(prompt, gold_img, gen_img, gold_code, gen_code, vis_query, max_tokens=2024):
    print("Querying...")
    
    # 注意：传入的 gold_img 和 gen_img 是纯 base64 字符串（不带 data header）
    # OpenAI 格式要求 image_url 必须包含前缀
    gold_data_url = f"data:image/png;base64,{gold_img}"
    gen_data_url = f"data:image/png;base64,{gen_img}"

    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=4096, # OpenAI SDK 参数通常是 max_tokens
            # temperature=0, # 建议加上，保持评测结果稳定
            messages=[
                # System Prompt 在 OpenAI 格式里是 messages 列表的第一条
                {
                    "role": "system", 
                    "content": "You are a helpful assistant evaluating visualization."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "This is the Visualization Query: "},
                        {"type": "text", "text": vis_query},
                        {"type": "text", "text": "This is the **ground-truth (reference)** image: "},
                        # OpenAI 图片格式
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": gold_data_url,
                                "detail": "high" # 显式指定高分辨率模式
                            }
                        },
                        {"type": "text", "text": "This is the code for the correct ground-truth image: "},
                        {"type": "text", "text": gold_code},
                        {"type": "text", "text": "This is the **under-test (assessed)** image:"},
                        # OpenAI 图片格式
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": gen_data_url,
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": "This is the code for the under-test generated image: "},
                        {"type": "text", "text": gen_code},
                    ]
                }
            ],
        )
        # OpenAI 的返回对象不是 response.content[0].text，而是这个路径
        return completion.choices[0].message.content
        
    except Exception as e:
        # 抛出异常让外层处理，或者在这里打印
        print(f"API Call Error: {e}")
        raise e


def extract_prompts(filename):
    prompts = []
    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if "viseval new prompt" in row:
                    prompts.append(row["viseval new prompt"])
    except Exception as e:
        print(f":x: Error reading prompts: {e}")
    return prompts


def do_vis_eval(all_queries):
    prompt_template = extract_prompts(prompts_csv)[0]
    new_queries = []
    for query in tqdm(all_queries):

        print(query['visualization_test'].keys())

        if not query['visualization_test']['vis_success'] or len(query['visualization_test']['gen_vis_list']) != 1:
            copy_query = copy.deepcopy(query)
            copy_query['visualization_llm_eval'] = {
                "rationale": "",
                "errors": "Crash" if not query['visualization_test']['vis_success'] else "VisFail",
            }
            new_queries.append(copy_query)
            continue

        try:
            # 注意：query_model 现在直接返回 text 字符串了，不需要再 .content[0].text
            raw_text = query_model(
                prompt_template,
                query["gt_visualization"].split('base64,')[1],
                query['visualization_test']['gen_vis_list'][0].split('base64,')[1],
                query["visualization_gt_code"],
                query["visualization_gen_code"],
                query["visualization_query"]
            )
            rationale, errors = safe_parse_json_from_claude(raw_text)
        except Exception as e:
            rationale = ""
            errors = f":x: Exception: {e}"
        copy_query = copy.deepcopy(query)
        copy_query['visualization_llm_eval'] = {
            "rationale": rationale,
            "errors": errors
        }
        new_queries.append(copy_query)
    return new_queries

