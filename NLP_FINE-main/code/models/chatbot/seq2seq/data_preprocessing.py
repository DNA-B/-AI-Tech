import pandas as pd
from tqdm import tqdm

train_df = pd.read_csv("/opt/ml/emotion_dialogue/train.csv")
valid_df = pd.read_csv("/opt/ml/emotion_dialogue/valid.csv")

new_train_df = pd.DataFrame(columns=['input_texts', 'labels'])
new_valid_df = pd.DataFrame(columns=['input_texts', 'labels'])

def basic_preprocessing(row, i):
    if (type(row[f"사람문장{i}"]) != float) and (type(row[f"시스템문장{i}"]) != float):
        if i != 1:
            new_row["input_texts"] += row[f"시스템문장{i-1}"] + '</s>'
        new_row["input_texts"] += row[f"사람문장{i}"] + '</s>'
        new_row["labels"] = row[f"시스템문장{i}"] + '</s>'     

        return pd.DataFrame(new_row, index=[0])
    else:
        return None

def no_multi_preprocessing(row, i):
    if (type(row[f"사람문장{i}"]) != float) and (type(row[f"시스템문장{i}"]) != float):
        new_row["input_texts"] = row[f"사람문장{i}"] + '</s>'
        new_row["labels"] = row[f"시스템문장{i}"] + '</s>'
        new_row["감정대분류"] = row['감정_대분류']
        new_row["감정소분류"] = row['감정_소분류']
        
        return pd.DataFrame(new_row, index=[0])
    else:
        return None
    
for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    new_row = {'감정대분류': '', '감정소분류': '', 'input_texts': '', 'labels': ''}
    
    for i in range(1, 4):
        new_row = no_multi_preprocessing(row, i)
        if new_row is not None:
          if new_row.loc[0, "labels"][-5:] == '?</s>':
            new_train_df = pd.concat([new_train_df, new_row], ignore_index=True)
            
new_train_df.to_csv("./data/emotion_dialogue/seq2seq_train/nomulti_emotion_question_train.csv")  

      
for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
    new_row = {'감정대분류': '', '감정소분류': '', 'input_texts': '', 'labels': ''}
    
    for i in range(1, 4):
        new_row = no_multi_preprocessing(row, i)
        if new_row is not None:
          if new_row.loc[0, "labels"][-5:] == '?</s>':
            new_valid_df = pd.concat([new_valid_df, new_row], ignore_index=True)
      
new_valid_df.to_csv("./data/emotion_dialogue/seq2seq_validation/nomulti_emotion_question_valid.csv")