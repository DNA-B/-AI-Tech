import torch
import pandas as pd
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, score, input_ids, attention_mask, token_type_ids):
        self.score = score
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
    
    def __getitem__(self, idx):
        item = {
            "input_ids" : self.input_ids[idx],
            "attention_mask" : self.attention_mask[idx],
            "token_type_ids" : self.token_type_ids[idx],
            "labels" : self.score[idx]
        }
        
        return item
    
    def __len__(self):
        return len(self.score)
            
    
def preprocessing(data, tokenizer):

    output = tokenizer(list(data['full_sen']),
                       return_tensors='pt',
                       padding=True,
                       max_length=1028,
                       truncation=True,
              )
    
    score = data['score']
    
    return output, score


def data_load(data_path, tokenizer):
    # data 불러오기
    raw = []

    with open(data_path, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            
            lst = line.split("\t")
            raw.append(lst)

    score, sen1, sen2 = [], [], []
    for lst in raw[1:]:
        score.append(float(lst[4]))
        sen1.append(lst[5])
        sen2.append(lst[6][:-1])
        
    df = pd.DataFrame({"score" : score,
                    "sen1" : sen1,
                    "sen2" : sen2
                    })
    
    df['full_sen'] = df['sen1'] + tokenizer.sep_token + df['sen2']
    
    return df

class KoBART_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            "input_ids" : self.input_ids[idx],
            "labels" : self.labels[idx]
        }

        return item

    def __len__(self):
        return len(self.labels)
    
def KoBART_data_load(df, tokenizer):
    
    conversations_input_ids = tokenizer(
        list(df['conversations']),
        return_tensors='pt',
        padding=True
    )['input_ids']
    
    score = tokenizer(
        list(df['summarizations']),
        return_tensors='pt',
        padding=True
    )['input_ids']
    
    return conversations_input_ids, score