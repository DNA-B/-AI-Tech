import torch, re
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd
from konlpy.tag import Mecab
from tqdm import tqdm

def eval(saved_model_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    add_special_tokens = {'additional_special_tokens' : ["[USER]", "[SYSTEM]"]}
    tokenizer.add_special_tokens(add_special_tokens)

    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)

    test_df = pd.read_csv("/opt/ml/code/models/summarization/kobart_test.csv")

    mecab = Mecab()

    device = torch.device("cuda")
    model.to(device)
    print(device)

    generated_text = []
    answer_text = []

    print("***generating***")
    for i in tqdm(range(len(test_df))):
        conversation = test_df.loc[i, "conversations"]
        conversation_ids = tokenizer(conversation, return_tensors='pt')["input_ids"].to(device)
        summary_ids = model.generate(conversation_ids, max_length=100)
        output_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        generated_text.append(output_text)
        answer_text.append(test_df.loc[i, 'summarizations'])
        

    rouge_one_score = []
    rouge_su_score = []

    print("****scoring****")
    for generated, answer in tqdm(zip(generated_text, answer_text)):
        generated = re.sub("<pad>","",generated)
        generated_mecab = mecab.morphs(generated)
        answer = re.sub("<pad>","",answer)
        answer_mecab = mecab.morphs(answer)
        
        if len(generated) != 0 :
            one_generated_mecab = set(generated_mecab)
            one_answer_mecab = set(answer_mecab)
            
            one_intersection = one_generated_mecab & one_answer_mecab
            
            one_recall = len(one_intersection) / len(one_answer_mecab)
            one_precision = len(one_intersection) / len(one_generated_mecab)
            
            one_f1 = 2 * one_precision * one_recall / (one_precision + one_recall)
            
            su_generated_mecab = one_generated_mecab
            
            if len(generated_mecab) == 1:
                continue
            elif len(generated_mecab) == 2:
                su_generated_mecab |= set(tuple(generated_mecab))
            else:
                for idx in range(len(generated_mecab) - 2):
                    for i in range(1,3):
                        su_generated_mecab.add((generated_mecab[idx] , generated_mecab[idx + i]))
                su_generated_mecab.add((generated_mecab[-2], generated_mecab[-1]))
            
            su_answer_mecab = one_answer_mecab
            
            if len(answer_mecab) == 1:
                continue
            elif len(answer_mecab) == 2:
                su_answer_mecab |= set(tuple(answer_mecab))
            else:
                for idx in range(len(answer_mecab) - 2):
                    for i in range(1,3):
                        su_answer_mecab.add((answer_mecab[idx] , answer_mecab[idx + i]))
                su_answer_mecab.add((answer_mecab[-2], answer_mecab[-1]))
                
            su_intersection = su_answer_mecab & su_generated_mecab
            
            su_recall = len(su_intersection) / len(su_answer_mecab)
            su_precision = len(su_intersection) / len(su_generated_mecab)
            
            su_f1 = 2 * su_precision * su_recall / (su_precision + su_recall)   
            
        else:
            one_f1 = 0
            su_f1 = 0
        
        rouge_one_score.append(one_f1)
        rouge_su_score.append(su_f1)
        
    rouge = {
        "rouge-1" : sum(rouge_one_score) / len(rouge_one_score),
        "rouge-su" : sum(rouge_su_score) / len(rouge_su_score)
    }

    print(rouge)

if __name__ == "__main__":
    eval("/opt/ml/code/models/summarization/save_folder/pytorch_model.bin")