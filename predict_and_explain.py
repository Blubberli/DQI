import shap
import transformers
# import nlp
import torch
import numpy as np
import scipy as sp
from transformers import RobertaConfig
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, Trainer, AutoConfig
import pandas as pd
from scipy.special import softmax

set_seed(42)


def _from_pretrained(cls, *args, **kw):
    """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
    try:
        return cls.from_pretrained(*args, **kw)
    except OSError as e:

        return cls.from_pretrained(*args, from_tf=True, **kw)


tokenizer = AutoTokenizer.from_pretrained('roberta-base')

model_path = "/Users/falkne/PycharmProjects/DQI/results/augmented_base/jlev/split3/checkpoint-52/pytorch_model.bin"

model_config = transformers.AutoConfig.from_pretrained(
    'roberta-base',
    num_labels=4,
    output_hidden_states=True,
    output_attentions=True,
)
# This is a just a regular PyTorch model.
model = _from_pretrained(
    transformers.AutoModelForSequenceClassification,
    'roberta-base',
    config=model_config)
print("model initialized")
checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
model.load_state_dict(checkpoint)
print("loaded model")


# define a prediction function
def f(x):
    texts = x["cleaned_comment"]
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in texts])
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = softmax(outputs, axis=1)
    #scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    #val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
    return scores


# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

training_data = pd.read_csv("/Users/falkne/PycharmProjects/DQI/data/5foldAugmentedEDA/jlev/split3/train.csv", sep="\t")
shap_values = explainer(training_data[:1])

print(shap_values)
shap.plots.text(shap_values[3])
shap.plots.bar(shap_values.abs.sum(0))
# explain the model's predictions on IMDB reviews
# shap_values = explainer(imdb_train[:10], fixed_context=1)
