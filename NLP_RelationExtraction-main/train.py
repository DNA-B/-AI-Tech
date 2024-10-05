import pickle as pickle
import pandas as pd
import torch
import numpy as np
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from module.load_data import *
import wandb
import yaml

from custom.CustomModel import *
from custom.CustomDataCollator import *
from custom.CustomTrainer import *

from module.seed_everything import seed_everything
from module.train_val_split import train_val_split
from module.add_token import *

from utils.compute_metrics import *
from utils.label_to_num import *

def train():
    with open('/opt/ml/module/config.yaml') as f:
        CFG = yaml.safe_load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    if CFG['PRE_TRAIN'] == 'None':
        MODEL_NAME = CFG['MODEL_NAME']
    else:
        MODEL_NAME = CFG['MODEL_PATH']
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = add_token(tokenizer, CFG['MODEL_TYPE'])

    if CFG['FOLD']:
        train_dataset = load_data(CFG['ENSEMBLE_TRAIN_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])
        dev_dataset = load_data(CFG['ENSEMBLE_DEV_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])

    else:
        dataset = load_data(CFG['TRAIN_PATH'], CFG['MODEL_TYPE'], CFG['DISCRIP'], CFG['DO_SEQUENTIALBERTMODEL'])
        train_dataset, dev_dataset = train_val_split(dataset, CFG['RATIO'])

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)
    
    if CFG['DO_SEQUENTIALBERTMODEL']:
        tokenized_train, entity_type_train, entity_words_train = sequentialdoublebert_tokenized_dataset(train_dataset, tokenizer, CFG['MODEL_TYPE'])
        tokenized_dev, entity_type_dev, entity_words_dev = sequentialdoublebert_tokenized_dataset(dev_dataset, tokenizer, CFG['MODEL_TYPE'])

        RE_train_dataset = RESequentialDoubleBERTDataset(tokenized_train, train_label, entity_type_train, entity_words_train)
        RE_dev_dataset = RESequentialDoubleBERTDataset(tokenized_dev, dev_label, entity_type_dev, entity_words_dev)

        model = SequentialDoubleBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer,
                                    model_type=CFG['MODEL_TYPE'], device=device)
        
        data_collator = SequentialDoubleBertDataCollator(tokenizer)
    
    elif CFG['MODEL_TYPE'] == 'base':
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

        RE_train_dataset = REDataset(tokenized_train, train_label)
        RE_dev_dataset = REDataset(tokenized_dev, dev_label)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        
        data_collator = DataCollatorWithPadding(tokenizer)

    elif CFG['MODEL_TYPE'] == 'entity_special':
        tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

        RE_train_dataset = RESpecialDataset(tokenized_train, train_label, entity_type_train)
        RE_dev_dataset = RESpecialDataset(tokenized_dev, dev_label, entity_type_dev)

        model = SpecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

        data_collator = CustomDataCollator(tokenizer)

    elif CFG['MODEL_TYPE'] == 'entity_punct' or CFG['MODEL_TYPE'] == 'ko_entity_punct':
        tokenized_train = punct_tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = punct_tokenized_dataset(dev_dataset, tokenizer)

        RE_train_dataset = REDataset(tokenized_train, train_label)
        RE_dev_dataset = REDataset(tokenized_dev, dev_label)

        model = SpecialPunctBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer)

    elif CFG['MODEL_TYPE'] == 'cls_entity_special':
        tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

        RE_train_dataset = RESpecialDataset(tokenized_train, train_label, entity_type_train)
        RE_dev_dataset = RESpecialDataset(tokenized_dev, dev_label, entity_type_dev)

        model = CLSSpecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer)

    elif CFG["MODEL_TYPE"] == 'no_cls_entity_special':
        tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

        RE_train_dataset = RESpecialDataset(tokenized_train, train_label, entity_type_train)
        RE_dev_dataset = RESpecialDataset(tokenized_dev, dev_label, entity_type_dev)

        model = NoCLSSpecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer)

    elif CFG["MODEL_TYPE"] == 'ko_entity_special':
        tokenized_train, entity_type_train = special_tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev, entity_type_dev = special_tokenized_dataset(dev_dataset, tokenizer)

        RE_train_dataset = RESpecialDataset(tokenized_train, train_label, entity_type_train)
        RE_dev_dataset = RESpecialDataset(tokenized_dev, dev_label, entity_type_dev)

        model = KoSpecialEntityBERT(MODEL_NAME, config=model_config, tokenizer=tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer)

    model.to(device)

    if CFG['SWEEP_AVAILABLE']:
        wandb.init()
        cfg = wandb.config
        wandb.run.name = f"{cfg.lr}, {cfg.epochs}"

        training_args = TrainingArguments(
            output_dir=CFG['OUTPUT_DIR'],
            save_total_limit=CFG['TOTAL_SAVE_MODEL'],
            save_steps=CFG['SAVING_STEP'],
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            warmup_steps=CFG['WARMUP_STEP'],
            weight_decay=CFG['WEIGHT_DECAY'],
            logging_dir=CFG['LOGGING_DIR'],
            logging_steps=CFG['LOGGING_STEP'],
            logging_strategy=CFG['STRATEGY'],
            save_strategy=CFG['STRATEGY'],
            evaluation_strategy=CFG['STRATEGY'],
            eval_steps=CFG['EVAL_STEP'],
            load_best_model_at_end=True,
            report_to="wandb",
            metric_for_best_model='micro f1 score')
    
    else:
        wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
        training_args = TrainingArguments(
            output_dir=CFG['OUTPUT_DIR'],
            save_total_limit=CFG['TOTAL_SAVE_MODEL'],
            save_steps=CFG['SAVING_STEP'],
            num_train_epochs=CFG['MAX_EPOCH'],
            learning_rate=CFG['LR'],
            per_device_train_batch_size=CFG['BATCH_SIZE'],
            per_device_eval_batch_size=CFG['BATCH_SIZE'],
            warmup_steps=CFG['WARMUP_STEP'],
            weight_decay=CFG['WEIGHT_DECAY'],
            logging_dir=CFG['LOGGING_DIR'],
            logging_steps=CFG['LOGGING_STEP'],
            logging_strategy=CFG['STRATEGY'],
            save_strategy=CFG['STRATEGY'],
            evaluation_strategy=CFG['STRATEGY'],
            eval_steps=CFG['EVAL_STEP'],
            load_best_model_at_end=True,
            report_to="wandb",
            metric_for_best_model='micro f1 score')

    trainer = CustomTrainer(
        loss_fn=CFG['LOSS_FN'],
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        eval_dataset=RE_dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    trainer.train()
    model.save_pretrained(CFG['MODEL_SAVE_DIR'])
    wandb.finish()

if __name__ == '__main__':
    seed_everything()
  
    with open('/opt/ml/module/config.yaml') as f:
        CFG = yaml.safe_load(f)
    
    if CFG['SWEEP_AVAILABLE']:
        sweep_configuration = CFG['SWEEP_CONFIGURATION']
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=CFG['WANDB_PROJECT'])
        wandb.agent(sweep_id=sweep_id, function=train, count=CFG['SWEEP_COUNT'])
  
    else:
        train()