import pandas as pd
import torch
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text):
        self.text = text
  
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.text.items()}
        return item
  
    def __len__(self):
        return len(self.text['input_ids'])


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    
    return data

def tokenized_dataset(data, tokenizer):
    with tqdm(total=len(data), desc="Tokenizing Text", unit="text") as pbar:
        tokenized_text = tokenizer(
            list(data['input_texts']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        pbar.update(len(data))
    
    return tokenized_text