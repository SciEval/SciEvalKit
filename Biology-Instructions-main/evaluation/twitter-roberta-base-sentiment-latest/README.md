---
language: en
widget:
- text: Covid cases are increasing fast!
datasets:
- tweet_eval
license: cc-by-4.0
---


# Twitter-roBERTa-base for Sentiment Analysis - UPDATED (2022)

This is a RoBERTa-base model trained on ~124M tweets from January 2018 to December 2021, and finetuned for sentiment analysis with the TweetEval benchmark. 
The original Twitter-based RoBERTa model can be found [here](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) and the original reference paper is [TweetEval](https://github.com/cardiffnlp/tweeteval). This model is suitable for English. 

- Reference Paper: [TimeLMs paper](https://arxiv.org/abs/2202.03829). 
- Git Repo: [TimeLMs official repository](https://github.com/cardiffnlp/timelms).

<b>Labels</b>: 
0 -> Negative;
1 -> Neutral;
2 -> Positive

This sentiment analysis model has been integrated into [TweetNLP](https://github.com/cardiffnlp/tweetnlp). You can access the demo [here](https://tweetnlp.org).

## Example Pipeline
```python
from transformers import pipeline
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("Covid cases are increasing fast!")
```
```
[{'label': 'Negative', 'score': 0.7236}]
```

## Full classification example

```python
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)
text = "Covid cases are increasing fast!"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# text = "Covid cases are increasing fast!"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)
# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
```

Output: 

```
1) Negative 0.7236
2) Neutral 0.2287
3) Positive 0.0477
```


### References 
```
@inproceedings{camacho-collados-etal-2022-tweetnlp,
    title = "{T}weet{NLP}: Cutting-Edge Natural Language Processing for Social Media",
    author = "Camacho-collados, Jose  and
      Rezaee, Kiamehr  and
      Riahi, Talayeh  and
      Ushio, Asahi  and
      Loureiro, Daniel  and
      Antypas, Dimosthenis  and
      Boisson, Joanne  and
      Espinosa Anke, Luis  and
      Liu, Fangyu  and
      Mart{\'\i}nez C{\'a}mara, Eugenio" and others,
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-demos.5",
    pages = "38--49"
}

```

```
@inproceedings{loureiro-etal-2022-timelms,
    title = "{T}ime{LM}s: Diachronic Language Models from {T}witter",
    author = "Loureiro, Daniel  and
      Barbieri, Francesco  and
      Neves, Leonardo  and
      Espinosa Anke, Luis  and
      Camacho-collados, Jose",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-demo.25",
    doi = "10.18653/v1/2022.acl-demo.25",
    pages = "251--260"
}

```