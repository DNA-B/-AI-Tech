import torch
import wandb
import yaml
import os
import numpy as np
import pandas as pd
import random
import argparse

from transformers import GPTNeoXForCausalLM, AutoTokenizer
from kullm.utils.prompter import Prompter

from seq2seq.compute_metrics import Evaluation_Metrics
from tqdm import tqdm


def infer(model, tokenizer, prompter, instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    input_ids = tokenizer(prompt, return_tensors='pt')
   
    output = model.generate(
        input_ids['input_ids'].to('cuda:0'),
        max_length=2048,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    s = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    result = prompter.get_response(s)

    return result, output.scores[len(input_ids['input_ids']):]

def inference(CFG):
    model = GPTNeoXForCausalLM.from_pretrained(
        CFG['MODEL_SAVE_DIR'],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(CFG['MODEL_NAME'])
    
    prompter = Prompter("kullm")
    evaluation = Evaluation_Metrics(tokenizer)
    
    wandb.init(project=CFG['WANDB_PROJECT'], name=CFG['WANDB_NAME'])
    
    test_data = pd.read_csv(CFG['TEST_PATH'])
    predictions = []
    labels = []
    perplexities = []
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        result, scores = infer(model=model, tokenizer=tokenizer, prompter=prompter, instruction="주어진 문장들은 이전 대화 내용들입니다. 이에 알맞은 시스템 응답을 만들어주세요.", 
                       input_text=row['input_texts'])
        
        probs = torch.tensor(1.0)
   
        for i in range(len(scores)):
            prob = torch.softmax(scores[i].detach().cpu().to(torch.float32), dim=-1)
            prob, index = torch.max(prob, dim=-1)
            probs = probs * prob
            
            if tokenizer.eos_token_id == index:
                break
        
        perplexity = (1 / probs) ** (1 / (i+1))
        perplexities.append(perplexity.item())
        labels.append(row['labels'])
        predictions.append(result)
    
    perplexity = sum(perplexities) / len(perplexities)    
    similarity = evaluation.calculate_similarity(predictions, labels)
    rouge = evaluation.calculate_rouge(predictions, labels)
    
    wandb.log({
        'rouge-su': rouge['rougeSU'],
        'rouge1': rouge['rouge1'],
        'similarity_scores': similarity,
        'perplexity': perplexity
    })
    
    test_data['prediction'] = predictions
    test_data.to_csv('/opt/ml/prediction/prediction.csv')
    
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