from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import yaml
import pandas as pd
from module.mlm_dataset import TextDatasetForMaskedLanguageModeling
from sklearn.model_selection import train_test_split

def MLM():
    with open('/opt/ml/module/config.yaml') as f:
        CFG = yaml.safe_load(f)
    
    model = AutoModelForMaskedLM.from_pretrained(CFG['MODEL_NAME'])
    tokenizer = AutoTokenizer.from_pretrained(CFG['MODEL_NAME'])

    train = pd.read_csv(CFG['TRAIN_PATH'])
    sentences = train['sentence']
    sentences = [sentence + '\n' for sentence in sentences]

    train, dev = train_test_split(sentences, test_size=CFG['RATIO'], random_state=14)
    
    with open('/opt/ml/dataset/train/mlm_train.txt', 'w', encoding='UTF8') as f:
        f.writelines(train)
    with open('/opt/ml/dataset/train/mlm_dev.txt', 'w', encoding='UTF8') as f:
        f.writelines(dev)
    
    train_dataset = TextDatasetForMaskedLanguageModeling(
    tokenizer=tokenizer,
    file_path='/opt/ml/dataset/train/mlm_train.txt',
    block_size=128,
    overwrite_cache=True,
    short_seq_probability=0.1,
    nsp_probability=0.0)

    dev_dataset = TextDatasetForMaskedLanguageModeling(
    tokenizer=tokenizer,
    file_path='/opt/ml/dataset/train/mlm_dev.txt',
    block_size=128,
    overwrite_cache=True,
    short_seq_probability=0.1,
    nsp_probability=0.0)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

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
        evaluation_strategy='steps',
        eval_steps=CFG['EVAL_STEP'],
        load_best_model_at_end=True,
        report_to="wandb")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset)

    trainer.train()
    model.save_pretrained(CFG['MODEL_SAVE_DIR'])

if __name__=='__main__':
    MLM()