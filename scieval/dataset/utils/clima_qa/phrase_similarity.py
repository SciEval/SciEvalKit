import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import re


class PhraseSimilarity():
    def __init__(self) -> None:
        print('Loading universal-sentence-encoder for phrase similarity')
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print('Model Loaded')

    def is_int(self, string):
        try:
            int(string)
            return True
        except:
            return False

    def is_float(self, string):
        try:
            float(string)
            return True
        except:
            return False

    def extract_around_mask(self, sentence, mask_token='<MASK>', num_words=4):
        words = sentence.split()
        for word in words:
            if mask_token in word:
                mask_index = words.index(word)
                break

        if mask_index is None:
            print("No <MASK> found")
            return

        start = max(mask_index - num_words, 0)
        end = min(mask_index + num_words + 1, len(words))

        phrase_arr = words[start:end]

        extracted_phrase = ' '.join(phrase_arr)
        return extracted_phrase

    def phrase_similarity(self, blank_statement: str, generated_term: str, correct_term: str) -> float:
        try:
            phrase = self.extract_around_mask(sentence=blank_statement)

            # if integer/float, keep numbers from 0-9
            if self.is_int(correct_term) or self.is_float(correct_term):
                ground_truth_phrase = phrase.replace("<MASK>", re.sub(r'[^0-9.\-%]', '', correct_term.lower()))
            else:
                ground_truth_phrase = phrase.replace("<MASK>", re.sub(r'[^a-zA-Z0-9% ]+', '', correct_term.lower()))

            # if not integer/float, keep relevant symbols such as ., /, %, -
            if self.is_int(generated_term) or self.is_float(generated_term):
                generated_phrase = phrase.replace("<MASK>", re.sub(r'[^0-9.\-%]', '', generated_term.lower()))
            else:
                generated_phrase = phrase.replace("<MASK>", re.sub(r'[^a-zA-Z0-9% ]+', '', generated_term.lower()))

            sim = cosine_similarity(self.model([ground_truth_phrase]).numpy(), self.model([generated_phrase]).numpy())
            sim = sim[0][0]
            sim = 2 * (sim - 0.5)
            return sim
        except Exception as e:
            print(str(e))