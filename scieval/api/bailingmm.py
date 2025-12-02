import base64
from scieval.smp import *
from scieval.api.base import BaseAPI
from scieval.dataset import DATASET_TYPE
from scieval.smp.vlm import encode_image_file_to_base64
import time


class bailingMMWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str,
                 retry: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via bailingMM API.'
        if key is None:
            key = os.environ.get('BAILINGMM_API_KEY', None)
        assert key is not None, ('Please set the API Key for bailingMM.')
        self.key = key
        self.headers = {"Content-Type": "application/json"}
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            encoded_string = str(base64.b64encode(image_file.read()), 'utf-8')
            return encoded_string

    def prepare_inputs(self, inputs):
        msgs = cp.deepcopy(inputs)
        content = []
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                pass
            else:
                try:
                    image_data = self.image_to_base64(msg['value'])
                except Exception as e:
                    if self.verbose:
                        self.logger.error(e)
                    image_data = ''
                msg['value'] = image_data
            content.append(msg)
        return content

    def generate_inner(self, inputs, **kwargs) -> tuple:
        # 1. Pre-process inputs
        inputs = [inputs] if isinstance(inputs, str) else inputs
        messages = self.prepare_inputs(inputs)

        # 2. Construct payload
        service_url = "https://bailingchat.alipay.com/api/proxy/eval/antgmm/completions"
        payload = {
            "structInput": json.dumps([{"role": "user", "content": messages}]),
            "sk": self.key,
            "model": self.model,
            "timeout": 180000
        }

        # 3. Execute request
        start = time.time()
        response = requests.post(service_url, headers=self.headers, json=payload)

        # 4. Error check (bubbles up to BaseAPI)
        response.raise_for_status()

        if self.verbose:
            self.logger.info(f'Request latency: {time.time() - start}')

        # 5. Parse response (Original Logic)
        output = response.json()
        answer = output['preds']['pred']

        if self.verbose:
            self.logger.info(f'inputs: {inputs}\nanswer: {answer}')

        return 0, answer, 'Succeeded! '

class bailingMMAPI(bailingMMWrapper):

    def generate(self, message, dataset=None,**kwargs):
        return super(bailingMMAPI, self).generate(message, dataset=dataset,**kwargs)
