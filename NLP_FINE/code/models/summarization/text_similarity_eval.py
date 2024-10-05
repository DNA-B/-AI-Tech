from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import torch
from load_data import data_load
from tqdm import tqdm

saved_model_path = "/opt/ml/code/models/summarization/output/model_list/pytorch_model-roberta_large.bin"
saved_config_path = "/opt/ml/code/models/summarization/output/model_list/config-roberta_large.json"

model_name = "klue/roberta-large"
config = AutoConfig.from_pretrained(saved_config_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name , config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

state_dict = torch.load(saved_model_path)
model.load_state_dict(state_dict)
model.eval()

data_path = "/opt/ml/code/models/summarization/sts-test.tsv"
df = data_load(data_path, tokenizer)

from tqdm import tqdm

pred_list = []
ans_list = []
for i in tqdm(range(len(df))):
    ans_list.append(df.loc[i, 'score'])
    input = tokenizer(df.loc[i, "full_sen"], return_tensors='pt')
    output = float(model(**input)['logits'])
    pred_list.append(output)

correlation_coefficient = np.corrcoef(ans_list, pred_list)[0, 1]

print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
