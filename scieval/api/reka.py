from scieval.smp import *
from scieval.api.base import BaseAPI
from time import sleep
import mimetypes


class Reka_Wrapper(BaseAPI):

    is_api: bool = True
    INTERLEAVE: bool = False

    def __init__(self,
                 model: str = 'reka-flash-20240226',
                 key: str = None,
                 retry: int = 10,
                 system_prompt: str = None,
                 verbose: bool = True,
                 temperature: float = 0,
                 max_tokens: int = 1024,
                 **kwargs):

        try:
            import reka
        except ImportError:
            raise ImportError('Please install reka by running "pip install reka-api"')

        self.model = model
        default_kwargs = dict(temperature=temperature, request_output_len=max_tokens)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('REKA_API_KEY', '')
        super().__init__(retry=retry, verbose=verbose, system_prompt=system_prompt, **kwargs)

    def generate_inner(self, inputs, **kwargs) -> tuple:
        import reka
        reka.API_KEY = self.key
        dataset = kwargs.pop('dataset', None)
        prompt, image_path = self.message_to_promptimg(inputs, dataset=dataset)
        image_b64 = encode_image_file_to_base64(image_path)
        stream = kwargs.pop('stream', False)

        # 1. Call SDK
        if stream:
            response_gen = reka.chat(
                model_name=self.model,
                human=prompt,
                media_url=f'data:image/jpeg;base64,{image_b64}',
                stream=True,
                **self.kwargs
            )
            answer = ""
            for chunk in response_gen:
                # Assuming chunk is text or dict with text
                answer += chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
            return 0, answer, 'Succeeded'
        else:
            response = reka.chat(
                model_name=self.model,
                human=prompt,
                media_url=f'data:image/jpeg;base64,{image_b64}',
                **self.kwargs
            )
            return 0, response['text'], response

class Reka(Reka_Wrapper):

    def generate(self, message, dataset=None,**kwargs):
        return super(Reka_Wrapper, self).generate(message,**kwargs)
