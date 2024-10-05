import torch
import wandb
import yaml
import os
import numpy as np
import random
import math
import argparse

from transformers import (
    AutoConfig,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from load_data import *
from compute_metrics import Evaluation_Metrics


def train(CFG):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = CFG['MODEL_NAME']
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME, 
                                                        eos_token='</s>', 
                                                        pad_token='<pad>')

    config = AutoConfig.from_pretrained(MODEL_NAME,
                                        eos_token=tokenizer.eos_token,
                                        pad_token=tokenizer.pad_token)
                                                        
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config).to(device)
    evaluation = Evaluation_Metrics(tokenizer)
    
    wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
    
    training_args = TrainingArguments(
        evaluation_strategy='steps',
        eval_steps=CFG['EVAL_STEP'],
        learning_rate=CFG['LR'],
        logging_dir='./logs',
        logging_steps=CFG['LOGGING_STEP'],
        # load_best_model_at_end=True,
        num_train_epochs=CFG['MAX_EPOCH'],
        output_dir='./results',
        per_device_train_batch_size=CFG['TRAIN_BATCH_SIZE'],
        per_device_eval_batch_size=CFG['EVAL_BATCH_SIZE'],
        save_strategy='steps',
        save_steps=CFG['SAVING_STEP'],
        save_total_limit=1,
        warmup_steps=CFG['WARMUP_STEP'],
        weight_decay=CFG['WEIGHT_DECAY'],
        report_to='wandb'
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_data = load_data(CFG['TRAIN_PATH'])
    dev_data = load_data(CFG['DEV_PATH'])
    
    tokenized_train = tokenized_dataset(train_data, tokenizer)
    tokenized_dev = tokenized_dataset(dev_data, tokenizer)
  
    train_dataset = Dataset(tokenized_train)
    dev_dataset = Dataset(tokenized_dev)
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=evaluation.compute_metrics,
        preprocess_logits_for_metrics=evaluation.preprocess_logits_for_metrics,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained(CFG['MODEL_SAVE_DIR'])
    
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
        
    train(CFG)