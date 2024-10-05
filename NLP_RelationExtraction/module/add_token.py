def add_token(tokenizer, model_type):
    if model_type == 'entity_special':
        tokenizer.add_special_tokens({"additional_special_tokens":['[S:PER]', '[/S:PER]', '[O:PER]', '[/O:PER]',
            '[S:ORG]', '[/S:ORG]', '[O:ORG]', '[/O:ORG]', '[O:POH]', '[/O:POH]',
            '[O:DAT]', '[/O:DAT]', '[S:LOC]', '[/S:LOC]', '[O:LOC]', '[/O:LOC]', '[O:NOH]', '[/O:NOH]']})
        
    elif model_type == 'entity_punct' or model_type == 'new_entity_punct':
        tokenizer.add_tokens(['§'])

    elif model_type == 'cls_entity_special':
        new_special_tokens = {"additional_special_tokens" : ['[SUBJ]' , '[OBJ]' , '[PER]' , '[ORG]',
            '[DAT]' , '[LOC]' , '[POH]' , '[NOH]']}
        tokenizer.add_special_tokens(new_special_tokens)
    
    return tokenizer