import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
add_special_tokens = {'additional_special_tokens' : ["[USER]", "[SYSTEM]"]}
tokenizer.add_special_tokens(add_special_tokens)

model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
model.resize_token_embeddings(len(tokenizer))

saved_model_path = "" # model path
state_dict = torch.load(saved_model_path)
model.load_state_dict(state_dict)

input = "" # 요약할 문장 ( 대화 데이터 )
input_ids = tokenizer(input, return_tensors='pt')['input_ids']
summary_ids = model.generate(input_ids, max_length=100)
output_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

print(output_text) # 이게 요약 문장입니다!