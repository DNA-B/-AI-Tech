from transformers import AutoTokenizer

def custom_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    add_special_tokens = {'sep_token' : "<sep>",
                        "cls_token" : "<cls>"}
    
    tokenizer.add_special_tokens(add_special_tokens)
    
    return tokenizer