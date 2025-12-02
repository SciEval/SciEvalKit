from ...smp import load_env
import os
from ...api import OpenAIWrapper

INTERNAL = os.environ.get('INTERNAL', 0)
MODLE_MAP = {
            'gpt-4-turbo': 'gpt-4-1106-preview',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0125': 'gpt-4-0125-preview',
            'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
            'chatgpt-1106': 'gpt-3.5-turbo-1106',
            'chatgpt-0125': 'gpt-3.5-turbo-0125',
            'gpt-4o': 'gpt-4o-2024-05-13',
            'gpt-4o-0806': 'gpt-4o-2024-08-06',
            'gpt-4o-1120': 'gpt-4o-2024-11-20',
            'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
            'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
            'deepseek': 'deepseek-ai/DeepSeek-V3',
            'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        }

def build_judge_model(**kwargs):
    from scieval.config import supported_VLM
    model_name = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    from ...smp import load_env
    load_env()

    model_name = os.environ.get('LOCAL_LLM', model_name)

    if 'class' in kwargs:
        cls_name = kwargs.pop('class')
        import scieval.api
        import scieval.vlm

        if hasattr(scieval.api, cls_name):
            cls = getattr(scieval.api, cls_name)
        elif hasattr(scieval.vlm, cls_name):
            cls = getattr(scieval.vlm, cls_name)
        else:
            raise ValueError(f"Judge class {cls_name} not found in api or vlm module.")

        return cls(model=model_name, **kwargs)

    model_name = MODLE_MAP[model_name] if model_name in MODLE_MAP else model_name
    if model_name in supported_VLM:
        try:
            return supported_VLM[model_name](**kwargs)
        except Exception as e:
            print(f"Failed to init judge model {model_name} from registry: {e}")
            raise e

    print(f"Model {model_name} not found in config, using OpenAIWrapper as fallback.")
    return OpenAIWrapper(model=model_name, **kwargs)


'''legacy method'''
def build_judge(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI, HFChatModel
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_version = MODLE_MAP[model] if model in MODLE_MAP else model
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    elif model == 'llama31-8b':
        model = HFChatModel(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from scieval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You cam see the specific error if the API call fails.
"""
