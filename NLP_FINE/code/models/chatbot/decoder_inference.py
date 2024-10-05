import torch
import wandb
import yaml
import os
import numpy as np
import random
import argparse

from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)

from seq2seq.load_data import *
from seq2seq.compute_metrics import Evaluation_Metrics
from torch.utils.data import DataLoader
from tqdm import tqdm



def inference(CFG):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = CFG['MODEL_NAME']
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
                                                        eos_token='</s>',
                                                        pad_token='<pad>',)
                                                        
    model = AutoModelForCausalLM.from_pretrained(CFG['MODEL_SAVE_DIR']).to(device)
    model.eval()
    evaluation = Evaluation_Metrics(tokenizer)

    wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
    
    test_data = load_data(CFG['TEST_PATH'])

    tokenized_test, tokenized_test_labels = tokenized_dataset(test_data, tokenizer, CFG['MAX_LENGTH'])
    
    test_dataset = Dataset(tokenized_test, tokenized_test_labels)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=data_collator)
    
    predictions = []
    labels = []
    decoded_predictions = []

    for test_features in tqdm(test_dataloader, total=len(test_dataloader)):
        test_features.to(device)
        inference_output = model.generate(
                test_features['input_ids'],
                max_length=256,
        )
            
        predictions += np.array(inference_output[:, test_features['input_ids'].size(-1):].detach().cpu()).tolist()
        labels += np.array(test_features['labels'].detach().cpu()).tolist()

        decoded_prediction = tokenizer.batch_decode(inference_output[:, test_features['input_ids'].size(-1):], skip_special_tokens=True)
        decoded_predictions += decoded_prediction

    pred_max_length = max(len(row) for row in predictions)
    predictions = np.array([row + [tokenizer.pad_token_id] * (pred_max_length - len(row)) for row in predictions])

    label_max_length = max(len(row) for row in labels)
    labels = np.array([row + [tokenizer.pad_token_id] * (label_max_length - len(row)) for row in labels])

    metric = evaluation.compute_metrics((predictions, labels))
    
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