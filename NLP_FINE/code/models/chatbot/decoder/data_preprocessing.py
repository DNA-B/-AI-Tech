import pandas as pd
from tqdm import tqdm

train_df = pd.read_csv("/opt/ml/emotion_dialogue/train.csv")
valid_df = pd.read_csv("/opt/ml/emotion_dialogue/valid.csv")

new_train_df = pd.DataFrame(columns=['input_texts'])
new_valid_df = pd.DataFrame(columns=['input_texts'])


for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    new_row = {'input_texts': ''}
    
    for i in range(1, 4):
        if (type(row[f"사람문장{i}"]) != float) and (type(row[f"시스템문장{i}"]) != float):
            new_row["input_texts"] += row[f"사람문장{i}"] + '</s>'
            new_row["input_texts"] += row[f"시스템문장{i}"] + '</s>'
            
    new_row = pd.DataFrame(new_row, index=[0])
    new_train_df = pd.concat([new_train_df, new_row], ignore_index=True)
    
new_train_df.to_csv("./data/emotion_dialogue/decoder_train/emotion_train.csv")  

      
for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
    new_row = {'input_texts': ''}
    
    for i in range(1, 4):
        if (type(row[f"사람문장{i}"]) != float) and (type(row[f"시스템문장{i}"]) != float):
            new_row["input_texts"] += row[f"사람문장{i}"] + '</s>'
            new_row["input_texts"] += row[f"시스템문장{i}"] + '</s>'
            
    new_row = pd.DataFrame(new_row, index=[0])
    new_valid_df = pd.concat([new_valid_df, new_row], ignore_index=True)
      
new_valid_df.to_csv("./data/emotion_dialogue/decoder_validation/emotion_valid.csv")