import pickle as pickle
import os
import pandas as pd
import torch

sub_type_dict = {'PER': '인물', 'ORG': '단체', 'LOC': '장소'}
obj_type_dict = {'PER': '인물', 'ORG': '단체', 'POH': '명칭', 'DAT': '날짜', 'LOC': '장소', 'NOH': '수'}

class REDataset(torch.utils.data.Dataset):
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
  
class RESpecialDataset(torch.utils.data.Dataset):
  def __init__(self, pair_dataset, labels, entity_type):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.entity_type = entity_type

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['subject_type'] = self.entity_type['subject_type'].iloc[idx]
    item['object_type'] = self.entity_type['object_type'].iloc[idx]
    return item

  def __len__(self):
    return len(self.labels)

class RESequentialDoubleBERTDataset(torch.utils.data.Dataset):
  def __init__(self, pair_dataset, labels, entity_type, entity_words):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.entity_type = entity_type
    self.entity_words = entity_words

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['subject_type'] = self.entity_type['subject_type'].iloc[idx]
    item['object_type'] = self.entity_type['object_type'].iloc[idx]
    item['subject_words'] = self.entity_words['subject_words'].iloc[idx]
    item['object_words'] = self.entity_words['object_words'].iloc[idx]
    return item

  def __len__(self):
    return len(self.labels)
  
def sequentialdoublebert_preprocessing_dataset(dataset, model_type):
  sentences = []
  subject_type = []
  object_type = []
  subject_words = []
  object_words = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)

    sub_word = f"' {sub_entity['word']} '"
    obj_word = f"' {sub_entity['word']} '"

    subject_words.append(sub_word)
    object_words.append(obj_word)

    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    subject_type.append(sub_type)
    object_type.append(obj_type)
      
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    if model_type == 'entity_special':
      sub_start, sub_end = f'[S:{sub_type}] ', f' [/S:{sub_type}]'
      obj_start, obj_end = f'[O:{obj_type}] ', f' [/O:{obj_type}]'

      if sub_idx[0] < obj_idx[0]:
        sentence = (sentence[:sub_idx[0]] + " " + sub_start + " " + sub_entity['word'] + " " + sub_end + " "
                  + sentence[sub_idx[1]+1:obj_idx[0]] + " " + obj_start + " " + obj_entity['word'] + " "
                  + obj_end + " " + sentence[obj_idx[1]+1:])
      else:
        sentence = (sentence[:obj_idx[0]] + " " + obj_start + " " + obj_entity['word'] + " " + obj_end + " "
                  + sentence[obj_idx[1]+1:sub_idx[0]] + " " + sub_start + " " + sub_entity['word'] + " "
                  + sub_end + " " + sentence[sub_idx[1]+1:])
    elif model_type == 'entity_punct':
      if sub_idx[0] < obj_idx[0]:
        sentence = (sentence[:sub_idx[0]] + f'@ § {sub_type} § ' + sub_entity['word'] + ' @'
                  + sentence[sub_idx[1]+1:obj_idx[0]] + f'# ^ {obj_type} ^ ' + obj_entity['word']
                  + ' #' + sentence[obj_idx[1]+1:])
      else:
        sentence = (sentence[:obj_idx[0]] + f'@ § {obj_type} § ' + obj_entity['word'] + ' @'
                  + sentence[obj_idx[1]+1:sub_idx[0]] + f'# ^ {sub_type} ^ ' + sub_entity['word']
                  + ' #' + sentence[sub_idx[1]+1:])
        
    sentences.append(sentence)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type, 'subject_words':subject_words, 'object_words':object_words, 'label':dataset['label']})

  return out_dataset

def add_discription(sentence, sub_word, obj_word, sub_type, obj_type, do_discrip=0):
  if do_discrip == 1:
    discription = f"이 문장에서{obj_word}는{sub_word}의{obj_type}이다."
    sentence += ':' + discription
  
  elif do_discrip == 2:
    discription = f" 이 문장에서 {sub_word}는 {sub_type}이고 {obj_word}는 {obj_type}이다."
    sentence += ":" + discription
   
  return sentence
  
def preprocessing_dataset(dataset, discrip):
  sentences = []
  subject_words = []
  object_words = []
  sentences = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_word = eval(sub_entity)['word']
    obj_word = eval(obj_entity)['word']
    sub_word = f' \'{sub_word}\' '
    obj_word = f' \'{obj_word}\' '
    
    subject_words.append(sub_word)
    object_words.append(obj_word)
    
    sentence = add_discription(sentence, sub_word, obj_word, eval(sub_entity)['type'], eval(obj_entity)['type'], discrip)
      
    sentences.append(sentence)
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'subject_entity':subject_words, 'object_entity':object_words, 'label':dataset['label'],})
  
  return out_dataset

def special_preprocessing_dataset(dataset, discrip):
  sentences = []
  subject_type = []
  object_type = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)

    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    subject_type.append(sub_type)
    object_type.append(obj_type)

    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_start, sub_end = f'[S:{sub_type}]', f'[/S:{sub_type}]'
    obj_start, obj_end = f'[O:{obj_type}]', f'[/O:{obj_type}]'
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "

    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + sub_start + sub_word + sub_end 
                  + sentence[sub_idx[1]+1:obj_idx[0]] + obj_start + obj_word
                  + obj_end + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + obj_start + obj_word + obj_end 
                  + sentence[obj_idx[1]+1:sub_idx[0]] + sub_start + sub_word
                  + sub_end + sentence[sub_idx[1]+1:])
    
    sentence = add_discription(sentence, sub_word, obj_word, f" \'{sub_type}\' ", f" \'{obj_type}\' ", discrip)
      
    sentences.append(sentence)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type})
  
  return out_dataset 

def cls_special_preprocessing_dataset(dataset):
  sentences = []
  subject_type = []
  object_type = []

  for subj_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    subj_entity = eval(subj_entity)
    obj_entity = eval(obj_entity)

    subj_type, obj_type = subj_entity['type'], obj_entity['type']
    subject_type.append(subj_type)
    object_type.append(obj_type)
    subj_idx , obj_idx = subj_entity['start_idx'], obj_entity['start_idx']

    if subj_idx < obj_idx:
      sentence = (sentence[:subj_idx] + '[SUBJ]' + subj_type + sentence[subj_idx:obj_idx] + '[OBJ]'
                  + obj_type + sentence[obj_idx:])
    else:
      sentence = (sentence[:obj_idx] + '[OBJ]' + obj_type + sentence[obj_idx:subj_idx] + '[SUBJ]'
                  + subj_type + sentence[subj_idx:])
      
    sentences.append(sentence)

  output_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type})
  
  return output_dataset

def punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@ §{sub_type}§' + sub_word + '@'
                  + sentence[sub_idx[1]+1:obj_idx[0]] + f'# ^{obj_type}^' + obj_word
                  + '#' + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + f'# ^{obj_type}^' + obj_word + '#'
                  + sentence[obj_idx[1]+1:sub_idx[0]] + f'@ §{sub_type}§' + sub_word
                  + '@' + sentence[sub_idx[1]+1:])
      # ex) 〈Something〉는 @ § PER § 조지 해리슨 @이 쓰고 # ^ ORG ^ 비틀즈 #가 1969년 앨범 《Abbey Road》에 담은 노래다
    
    sentence = add_discription(sentence, sub_word, obj_word, f" \'{sub_type}\' ", f" \'{obj_type}\' ", discrip)
      
    sentences.append(sentence)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'],})
  
  return out_dataset

def ko_punct_preprocessing_dataset(dataset, discrip):
  sentences = []
  
  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)
    
    sub_idx, obj_idx = [sub_entity['start_idx'], sub_entity['end_idx']], [obj_entity['start_idx'], obj_entity['end_idx']]
    sub_type, obj_type = sub_entity['type'], obj_entity['type']
    sub_word, obj_word = f" \'{sub_entity['word']}\' ", f" \'{obj_entity['word']}\' "
    
    if sub_idx[0] < obj_idx[0]:
      sentence = (sentence[:sub_idx[0]] + f'@ § {sub_type_dict[sub_type]} §{sub_word}@' + sentence[sub_idx[1]+1:obj_idx[0]]
                  + f'# ^ {obj_type_dict[obj_type]} ^{obj_word}#' + sentence[obj_idx[1]+1:])
    else:
      sentence = (sentence[:obj_idx[0]] + f'# ^ {obj_type_dict[obj_type]} ^{obj_word}#' + sentence[obj_idx[1]+1:sub_idx[0]]
                  + f'@ § {sub_type_dict[sub_type]} §{sub_word}@' + sentence[sub_idx[1]+1:])

    sentence = add_discription(sentence, sub_word, obj_word, f" \'{obj_type}\' ", f" \'{obj_type}\' ", discrip)
      
    sentences.append(sentence)
  
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'],})
  
  return out_dataset  

def ko_special_preprocessing_dataset(dataset,discrip):
  sentences = []
  subject_type = []
  object_type = []

  for sub_entity, obj_entity, sentence in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub_entity = eval(sub_entity)
    obj_entity = eval(obj_entity)

    start_subj = sub_entity['start_idx']
    start_obj = obj_entity['start_idx']
    sub_word = sub_entity['type']
    obj_word = obj_entity['type']
    sub_real_word = sub_entity['word']
    obj_real_word = obj_entity['word']

    subject_type.append(sub_word)
    object_type.append(obj_word)

    if start_subj < start_obj:
      sentence = (sentence[:start_subj] + "[SUBJ] " + obj_type_dict[sub_word] + " "
      + sentence[start_subj:start_obj] + "[OBJ] " + obj_type_dict[obj_word] + " " + sentence[start_obj:])

    else:
      sentence = (sentence[:start_obj] + "[OBJ] " + obj_type_dict[obj_word] + " "
      + sentence[start_obj:start_subj] + "[SUBJ] " + obj_type_dict[sub_word] + " " + sentence[start_subj:])
    
    sentence = add_discription(sentence, sub_real_word, obj_real_word, f" \'{sub_word}\' ", f" \'{obj_word}\' ", discrip)

    sentences.append(sentence)
  
  output_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'label':dataset['label'], 'subject_type':subject_type, 'object_type':object_type})
  
  return output_dataset

def load_data(dataset_dir, model_type, discrip, do_sequentialdoublebert=0):
  pd_dataset = pd.read_csv(dataset_dir)
  
  if do_sequentialdoublebert:
    dataset = sequentialdoublebert_preprocessing_dataset(pd_dataset, model_type)
  elif model_type == 'entity_special':
    dataset = special_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'entity_punct':
    dataset = punct_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'ko_entity_punct':
    dataset = ko_punct_preprocessing_dataset(pd_dataset, discrip)
  elif model_type == 'cls_entity_special' or model_type == "no_cls_entity_special":
    dataset = cls_special_preprocessing_dataset(pd_dataset)
  elif model_type == 'ko_entity_special':
    dataset = ko_special_preprocessing_dataset(pd_dataset, discrip)
  else:
    dataset = preprocessing_dataset(pd_dataset, discrip)

  return dataset

def tokenized_dataset(dataset, tokenizer):
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True)
  
  return tokenized_sentences

def special_tokenized_dataset(dataset, tokenizer):
  tokenized_sentences = tokenizer(
    list(dataset['sentence']),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    add_special_tokens=True)
  
  return tokenized_sentences, dataset[['subject_type', 'object_type']]

def punct_tokenized_dataset(dataset, tokenizer):
  tokenized_sentences = tokenizer(
    list(dataset['sentence']),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    add_special_tokens=True)
  
  return tokenized_sentences

def sequentialdoublebert_tokenized_dataset(dataset, tokenizer, model_type):
  if model_type == 'base':
    concat_entity = []
    for e01, e02 in zip(dataset['subject_words'], dataset['object_words']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)
    
    tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True)
    
  else:
    tokenized_sentences = tokenizer(
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True)

  return tokenized_sentences, dataset[['subject_type', 'object_type']], dataset[['subject_words', 'object_words']]