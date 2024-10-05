import numpy as np
import re
from konlpy.tag import Mecab

class compute_metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mecab = Mecab()
    
    def compute_rouge(self, preds):
        input_ids = np.argmax(preds.predictions[0], axis=2)
        labels = preds.label_ids
        
        generated_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        answer_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_one_score = []
        rouge_su_score = []
        
        for generated, answer in zip(generated_text, answer_text):
            generated = re.sub("<pad>","",generated)
            generated_mecab = self.mecab.morphs(generated)
            answer = re.sub("<pad>","",answer)
            answer_mecab = self.mecab.morphs(answer)
            
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

        return rouge
        
        