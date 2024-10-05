import torch
from sentence_transformers import SentenceTransformer, util
from konlpy.tag import Mecab
import numpy as np

class Evaluation_Metrics:
    def __init__(self, tokenizer):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.similarity_model = SentenceTransformer('jhgan/ko-sbert-sts').to(self.device)
        self.mecab = Mecab()
        
    def calculate_similarity(self, decoded_preds, decoded_labels):
        preds_vectors = self.similarity_model.encode(decoded_preds)
        labels_vectors = self.similarity_model.encode(decoded_labels)

        similarities = util.cos_sim(preds_vectors, labels_vectors)
        similarities = torch.diagonal(similarities, 0)
        similarity_score = torch.mean(similarities)
        
        return similarity_score * 100
            

    def calculate_rouge(self, decoded_preds, decoded_labels, max_skip=2):
        # rouge-su, rouge-1 계산
        rouge_su_scores = []
        rouge_one_scores = []
        
        for prediction, label in zip(decoded_preds, decoded_labels):
            prediction_tokens = self.mecab.morphs(prediction)
            reference_tokens = self.mecab.morphs(label)
            
            if len(prediction_tokens) != 0:
                prediction_skip_bigrams, reference_skip_bigrams = [], []
                for skip in range(max_skip+1):
                    prediction_skip_bigrams += [(prediction_tokens[i], prediction_tokens[i + skip]) for i in range(len(prediction_tokens) - skip)]
                    reference_skip_bigrams += [(reference_tokens[i], reference_tokens[i + skip]) for i in range(len(reference_tokens) - skip)]

                common_su_bigram = set(reference_skip_bigrams) & set(prediction_skip_bigrams)
                rouge_su_r = len(common_su_bigram) / len(reference_skip_bigrams)
                rouge_su_p = len(common_su_bigram) / len(prediction_skip_bigrams)
                rouge_su_score = 2 * (rouge_su_r * rouge_su_p) / (rouge_su_r + rouge_su_p) if (rouge_su_r != 0) or (rouge_su_p != 0) else 0

                common_unigram = set(reference_tokens) & set(prediction_tokens)
                rouge_one_r = len(common_unigram) / len(reference_tokens)
                rouge_one_p = len(common_unigram) / len(prediction_tokens)
                rouge_one_score = 2 * (rouge_one_r * rouge_one_p) / (rouge_one_r + rouge_one_p) if (rouge_one_r != 0) or (rouge_one_p != 0) else 0
                
            else:
                rouge_su_score = 0
                rouge_one_score = 0
                          
            rouge_su_scores.append(rouge_su_score)
            rouge_one_scores.append(rouge_one_score)
            
        rouge_scores = {}
        rouge_scores['rougeSU'] = sum(rouge_su_scores) / len(rouge_su_scores) * 100
        rouge_scores['rouge1'] = sum(rouge_one_scores) / len(rouge_one_scores) * 100

        return rouge_scores
    
    def compute_metrics(self, preds):
        predictions, labels = preds  
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge_scores = self.calculate_rouge(decoded_preds, decoded_labels)
        similarity_scores = self.calculate_similarity(decoded_preds, decoded_labels)
    
        return {
            'rouge-su': rouge_scores['rougeSU'],
            'rouge1': rouge_scores['rouge1'],
            'similarity_scores': similarity_scores
        }

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        logits (batch, max_seq_length, vocab) size의 tensor를 다 저장하면 memory 부족
        (batch, max_seq_length, vocab) -> (batch, max_seq_length)
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids # preds.predictions에 전달되는 값   