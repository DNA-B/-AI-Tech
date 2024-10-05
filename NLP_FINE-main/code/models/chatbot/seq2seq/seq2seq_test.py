# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
from konlpy.tag import Mecab

def calculate_similarity(sentence1, sentence2):
    sentence1_vectors = similarity_model.encode(sentence1)
    sentence2_vectors = similarity_model.encode(sentence2)
    similarity = util.cos_sim(sentence1_vectors, sentence2_vectors).item()
    
    return similarity*100

def calculate_rouge(sentence1, sentence2, max_skip=2):
    sentence1_tokens = mecab.morphs(sentence1)
    sentence2_tokens = mecab.morphs(sentence2)
    
    common_unigram = set(sentence1_tokens) & set(sentence2_tokens)
    rouge_one_r = len(common_unigram) / len(sentence2_tokens)
    rouge_one_p = len(common_unigram) / len(sentence1_tokens)
    # rouge_one_score = 2 * (rouge_one_r * rouge_one_p) / (rouge_one_r + rouge_one_p)
    
    return rouge_one_p*100, rouge_one_r*100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

model = AutoModelForSeq2SeqLM.from_pretrained("./best_model") # fine-tuning 모델 경로
model.to(device)

similarity_model = SentenceTransformer('jhgan/ko-sbert-sts').to(device)
mecab = Mecab()
num = 0

print("system: 안녕하세요! 오늘 하루는 어떠셨나요? 무슨 재미있는 일이 있었나요?")
chat_history = tokenizer.encode('system: 안녕하세요! 오늘 하루는 어떠셨나요? 무슨 재미있는 일이 있었나요?', return_tensors='pt').to(device)
before_bot_response = 'system: 안녕하세요! 오늘 하루는 어떠셨나요? 무슨 재미있는 일이 있었나요?'

while True:
    input_text = input('>> User: ')
    
    if input_text == "bye":
        break
    
    encoded_vector = tokenizer.encode('user: ' + input_text, return_tensors='pt').to(device)
    
    if num == 0:
        chat_history = torch.cat([chat_history, encoded_vector], dim=-1)
    else:
        chat_history = torch.cat([chat_history, bot_response, encoded_vector], dim=-1)
        
    if chat_history.shape[-1] > 256:
        chat_history = chat_history[:,-256:]
        
    with torch.no_grad():
        output = model.generate(
            chat_history,
            max_length=50,
            num_beams=5,
            top_k=20,
            no_repeat_ngram_size=2,
            length_penalty=0.65,
            repetition_penalty=2.0,
        )
        
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    
    similarity = calculate_similarity(decoded_output, before_bot_response)
    rouge_p, rouge_r = calculate_rouge(decoded_output, before_bot_response)
    
    # chat history를 사용하면 과거 했던 응답을 그대로 하는 경우가 있어서 이전 응답과 너무 비슷할 경우,
    # chat history를 보지 않고 현재 user 입력만 가지고 응답 생성
    if (rouge_p) > 70 or (rouge_r > 70) or (similarity > 85):
        output = model.generate(
            encoded_vector,
            max_length=50,
            num_beams=5,
            top_k=20,
            no_repeat_ngram_size=4,
            length_penalty=0.65,
            repetition_penalty=2.0,
        )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    if len(input_text) <= 6:
        decoded_output = 'system: 오늘 더 특별한 일은 없으셨나요? 더 자세히 말씀해 주세요!'
            
    print(decoded_output)
    before_bot_response = decoded_output
    bot_response = tokenizer.encode(decoded_output, return_tensors='pt').to(device)
    num += 1