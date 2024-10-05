import torch
import wandb
import yaml
import os
import numpy as np
import random
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

from seq2seq.load_data import *
from seq2seq.compute_metrics import Evaluation_Metrics
from torch.utils.data import DataLoader
from tqdm import tqdm



def inference(CFG):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = CFG['MODEL_NAME']
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG['MODEL_SAVE_DIR']).to(device)
    model.eval()
    
    evaluation = Evaluation_Metrics(tokenizer)
    
    wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
    
    test_data = load_data(CFG['TEST_PATH'])
    
    tokenized_test, tokenized_test_labels = tokenized_dataset(test_data, tokenizer, CFG['MAX_LENGTH'], add_eos=True)

    test_dataset = Dataset(tokenized_test, tokenized_test_labels)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=data_collator)
    
    predictions = []
    labels = []
    decoded_predictions = []
    perplexities = []
    
    for test_features in tqdm(test_dataloader, total=len(test_dataloader)):
        test_features.to(device)
        inference_output = model.generate(
                test_features['input_ids'],
                max_length=256,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        eos_value = torch.tensor([1 for _ in range(len(test_features['input_ids']))]) # batch size
        probs = torch.tensor([[1.0, 0.0] for _ in range(len(test_features['input_ids']))]) # batch size, 2
        
        for i in range(len(inference_output.scores)):
            prob = torch.softmax(inference_output.scores[i].detach().cpu(), dim=-1) # batch size, vocab size
            prob, index = torch.max(prob, dim=-1) # batch size
            
            prob[eos_value == 0] = 1.0
            probs[:, 0] = probs[:, 0] * prob
            probs[:, 1] = probs[:, 1] + eos_value
            
            mask = index == tokenizer.eos_token_id
            eos_value[mask] = 0

        perplexity = ((1 / probs[:, 0]) ** (1 / probs[:, 1])).mean()
        perplexities.append(perplexity.item())
        
        predictions += np.array(inference_output.sequences.detach().cpu()).tolist()
        labels += np.array(test_features['labels'].detach().cpu()).tolist()

        decoded_prediction = tokenizer.batch_decode(inference_output.sequences, skip_special_tokens=True)
        decoded_predictions += decoded_prediction
        
    pred_max_length = max(len(row) for row in predictions)
    predictions = np.array([row + [tokenizer.pad_token_id] * (pred_max_length - len(row)) for row in predictions])

    label_max_length = max(len(row) for row in labels)
    labels = np.array([row + [tokenizer.pad_token_id] * (label_max_length - len(row)) for row in labels])

    metric = evaluation.compute_metrics((predictions, labels))
    metric['perplexity'] = sum(perplexities) / len(perplexities)
    
    wandb.log(metric)
    output = pd.read_csv(CFG['TEST_PATH'])
    output['prediction'] = decoded_predictions
    output.to_csv('./prediction/prediction.csv') 
    
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--args_path", default=f"/opt/ml/code/models/chatbot/config.yaml", type=str, help=""
    )
    arg = parser.parse_args()

    with open(arg.args_path) as f:
        CFG = yaml.safe_load(f)
        
    random.seed(CFG['SEED'])
    np.random.seed(CFG['SEED'])
    os.environ["PYTHONHASHSEED"] = str(CFG['SEED'])
    torch.manual_seed(CFG['SEED'])
    torch.cuda.manual_seed(CFG['SEED'])  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True
        
    inference(CFG)