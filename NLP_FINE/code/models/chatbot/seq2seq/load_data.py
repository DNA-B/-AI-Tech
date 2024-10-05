import pandas as pd
import torch
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
  def __init__(self, text, labels):
    self.text = text
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.text.items()}
    item['labels'] = self.labels[idx]
    return item

  def __len__(self):
    return len(self.labels)


def load_data(data_dir):
    data_df = pd.read_csv(data_dir)

    return data_df

def tokenized_dataset(data, tokenizer, max_length, add_eos=True):
    with tqdm(total=len(data), desc="Tokenizing Text", unit="text") as pbar:
        tokenized_text = tokenizer(
          list(data['input_texts']),
        )
        pbar.update(len(data))
    
        for i in range(len(tokenized_text['input_ids'])):
            if len(tokenized_text['input_ids'][i]) > max_length:
                for key, val in tokenized_text.items():
                    tokenized_text[key][i] = val[i][-max_length:]
            
            if add_eos:
                if tokenized_text['input_ids'][i][-1] != tokenizer.eos_token_id:
                    tokenized_text['input_ids'][i].append(tokenizer.eos_token_id)
            
    with tqdm(total=len(data), desc="Tokenizing Labels", unit="text") as pbar:
        with tokenizer.as_target_tokenizer():
            tokenized_labels = tokenizer(
              list(data['labels']),
            )
        pbar.update(len(data))   
        
        for i in range(len(tokenized_labels['input_ids'])):
            if len(tokenized_labels['input_ids'][i]) > max_length:
                for key, val in tokenized_labels.items():
                    tokenized_labels[key][i] = val[i][-max_length:]
                    
            if tokenized_labels['input_ids'][i][-1] != tokenizer.eos_token_id:
                tokenized_labels['input_ids'][i].append(tokenizer.eos_token_id)
      
    return tokenized_text, tokenized_labels['input_ids']