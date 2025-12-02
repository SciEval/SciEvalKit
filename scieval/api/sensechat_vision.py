import json
import os
import string
import time
from typing import Optional
import pandas as pd
import requests
from scieval.smp import (
    LMUDataRoot,
    osp,
    read_ok,
    decode_base64_to_image_file,
    toliststr,
    listinstr,
    cn_string,
)
from scieval.api.base import BaseAPI
from scieval.dataset import img_root_map
from scieval.dataset import DATASET_TYPE


class SenseChatVisionWrapper(BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        base_url: str = "https://api.sensenova.cn/v1/llm/chat-completions",
        api_key: str = None,
        model: str = "SenseNova-V6-5-Pro",
        retry: int = 5,
        wait: int = 5,
        verbose: bool = True,
        system_prompt: str = None,
        max_tokens: int = 16384,
        **kwargs,
    ):
        self.base_url = base_url
        self.model = model
        self.fail_msg = "Failed to obtain answer via API. "
        self.api_key = os.getenv("SENSENOVA_API_KEY", api_key)
        assert self.api_key is not None, (
            "Please set the `SENSENOVA_API_KEY` environment variable or pass `api_key` in the config.json."
        )
        self.max_new_tokens = max_tokens
        self.thinking = False
        super().__init__(
            wait=wait,
            retry=retry,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)
        img_root = osp.join(ROOT, "images", img_root_map(dataset))
        os.makedirs(img_root, exist_ok=True)
        if "image" in line:
            if isinstance(line["image"], list):
                tgt_path = []
                assert "image_path" in line
                for img, im_name in zip(line["image"], line["image_path"]):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line["image"], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert "image_path" in line
            tgt_path = toliststr(line["image_path"])

        return tgt_path

    def image_to_base64(self, image_path):
        import base64

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

    def use_custom_prompt(self, *args, **kwargs):
        """Check if the prompt is customized."""
        return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        return prompt

    def build_mcq_cot_prompt(self, line, prompt):
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = {
            'multiple-choice': "You are an expert in {}. Please solve the university-level {} examination question, which includes interleaved images and text. Answer the preceding multiple choice question. The last line of your response should follow this format: 'Answer: \\boxed LETTER', where LETTER is one of the options. If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Think step by step logically, considering all relevant information before answering.",  # noqa: E501
            'open': 'You are an expert in {}. Please solve the university-level {} examination question, which includes interleaved images and text. Your output should be divided into two parts: First, reason about the correct answer. Then write the answer in the following format where X is only the answer and nothing else: "ANSWER: X"'  # noqa: E501
        }
        subject = '_'.join(line['id'].split('_')[1:-1])
        prompt = prompt[line['question_type']].format(subject, subject) + '\n' + question
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(["MME"], dataset):
            question = line["question"]
            prompt = question + " Answer the question using a single word or phrase."
        elif dataset is not None and listinstr(["HallusionBench"], dataset):
            question = line["question"]
            prompt = (
                question
                + " Please answer yes or no. Answer the question using a single word or phrase."
            )
        elif dataset is not None and DATASET_TYPE(dataset) == "MCQ":
            prompt = self.build_multi_choice_prompt(line, dataset)
            if "MMMU" in dataset:
                prompt = self.build_mcq_cot_prompt(line, prompt)
                self.thinking = True
        elif dataset is not None and DATASET_TYPE(dataset) == "VQA":
            if "MathVista" in dataset:
                prompt = line["question"]
                self.thinking = True
            elif listinstr(["LLaVABench"], dataset):
                question = line["question"]
                prompt = question + "\nAnswer this question in detail."
            elif listinstr(["MMVet"], dataset):
                prompt = line["question"]
            else:
                question = line["question"]
                prompt = (
                    question
                    + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                )
        else:
            prompt = line["question"]

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])

        return message

    def message_to_promptimg(self, message, dataset=None):
        if dataset is None or listinstr(["MMMU", "BLINK"], dataset):
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            image = [[x["value"] for x in message if x["type"] == "image"][0]]
        else:
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            image = [x["value"] for x in message if x["type"] == "image"]
        return prompt, image

    def set_max_num(self, dataset: Optional[str] = None) -> None:
        """Set the max_num based on the dataset."""
        if dataset is not None and listinstr(
            [
                "ChartQA_TEST",
                "MMMU_DEV_VAL",
                "MMMU_TEST",
                "MME-RealWorld",
                "VCR_EN",
                "VCR_ZH",
                "OCRVQA",
            ],
            dataset,
        ):
            self.max_num = 12
        elif dataset is not None and listinstr(
            ["DocVQA_VAL", "DocVQA_TEST", "DUDE", "MMLongBench_DOC", "SLIDEVQA"],
            dataset,
        ):
            self.max_num = 18
        elif dataset is not None and listinstr(
            ["InfoVQA_VAL", "InfoVQA_TEST", "OCRBench", "HRBench4K", "HRBench8K"],
            dataset,
        ):
            self.max_num = 24
        else:
            self.max_num = 6

    def generate_inner(self, inputs, **kwargs) -> tuple:
        # 1. Prepare
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get("dataset", None)
        stream = kwargs.pop('stream', False)

        self.set_max_num(dataset=dataset)
        prompt, image = self.message_to_promptimg(message=inputs, dataset=dataset)

        content = [{"image_base64": self.image_to_base64(item), "type": "image_base64"} for item in image]
        content.append({"text": prompt, "type": "text"})

        data = {
            "messages": [{"content": content, "role": "user"}],
            "max_new_tokens": self.max_new_tokens,
            "model": self.model,
            "stream": stream,
            "image_split_count": self.max_num,
            "thinking": {"enabled": self.thinking},
        }
        headers = {"Content-type": "application/json", "Authorization": self.api_key}

        # 2. Request
        response = requests.post(
            self.base_url, headers=headers, json=data, stream=stream
        )
        request_id = response.headers.get("x-request-id", "")
        self.logger.info(f"Request-id: {request_id}")
        time.sleep(1)

        # 3. Parse
        response.raise_for_status()  # Check 200

        if stream:
            answer = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        if '[DONE]' in decoded: break
                        try:
                            # Assuming standard SSE format, adjust if SenseChat differs
                            chunk = json.loads(decoded[6:])
                            answer += chunk["choices"][0]["delta"].get("content", "")
                        except:
                            pass
        else:
            resp_json = response.json()
            if "error" in resp_json:
                raise RuntimeError(resp_json["error"]["message"])
            answer = resp_json["data"]["choices"][0]["message"].strip()

        if self.verbose:
            self.logger.info(f"inputs: {inputs}\nanswer: {answer}")

        return 0, answer, "Succeeded! "


class SenseChatVisionAPI(SenseChatVisionWrapper):
    def generate(self, message, dataset=None,**kwargs):
        return super(SenseChatVisionAPI, self).generate(message, dataset=dataset,**kwargs)
