from transformers import AutoModel, BertPreTrainedModel, AutoModelForSequenceClassification, AutoConfig
import torch
from copy import deepcopy
import numpy as np

class SpecialEntityBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)
        
        self.calculated_token_count = 4
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        sub_tokens = ['[S:PER]', '[S:ORG]', '[S:LOC]', '[/S:PER]', '[/S:ORG]', '[/S:LOC]']
        token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        self.sub_entity_token = dict(zip(sub_tokens, token_ids))

        obj_tokens = ['[O:PER]', '[O:ORG]', '[O:POH]', '[O:LOC]', '[O:NOH]', '[O:DAT]',
                      '[/O:PER]', '[/O:ORG]', '[/O:POH]', '[/O:LOC]', '[/O:NOH]', '[/O:DAT]']
        token_ids = tokenizer.convert_tokens_to_ids(obj_tokens)
        self.obj_entity_token = dict(zip(obj_tokens, token_ids))
        
        self.classifier = torch.nn.Sequential(
            self.model.pooler,
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True))
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(self.calculated_token_count)])
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[1/self.calculated_token_count]] for _ in range(self.calculated_token_count)]))
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state
                        
        batch_size = len(input_ids)
        special_idx = list()
 
        for i in range(batch_size):
            
            sub_start, sub_end = self.sub_entity_token[f'[S:{subject_type[i]}]'], self.sub_entity_token[f'[/S:{subject_type[i]}]']
            obj_start, obj_end = self.obj_entity_token[f'[O:{object_type[i]}]'], self.obj_entity_token[f'[/O:{object_type[i]}]']
            
            sub_start_idx = torch.nonzero(input_ids[i] == sub_start)[0][0] 
            sub_end_idx = torch.nonzero(input_ids[i] == sub_end)[0][0]
            obj_start_idx = torch.nonzero(input_ids[i] == obj_start)[0][0]
            obj_end_idx = torch.nonzero(input_ids[i] == obj_end)[0][0]
            
            special_idx.append([sub_start_idx, sub_end_idx, obj_start_idx, obj_end_idx])
     
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(self.calculated_token_count)]

        logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(self.calculated_token_count)], dim=0)
        logits = torch.sum(self.weight_parameter * logits, dim=0)

        loss = None
        
        if labels is not None: 
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        if output_attentions:    
            outputs = {"loss": loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss": loss, "logits": logits}
        
        return outputs
        
class SpecialPunctBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)

        self.calculated_token_count = 2
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config
        
        ids = tokenizer.convert_tokens_to_ids(['@', '#'])
        self.sub_ids, self.obj_ids = ids[0], ids[1]
        
        self.classifier = torch.nn.Sequential(
            self.model.pooler,
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True))
        
        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(self.calculated_token_count)])
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[1/self.calculated_token_count]] for _ in range(self.calculated_token_count)]))
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state
                        
        batch_size = len(input_ids)
        special_idx = list()
        
        for i in range(batch_size):
            sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0]
            obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            
            special_idx.append([sub_start_idx, obj_start_idx])
        
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j],:] for i in range(batch_size)]) for j in range(self.calculated_token_count)]

        logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(self.calculated_token_count)], dim=0)
        logits = torch.sum(self.weight_parameter * logits, dim=0) 
    
        loss = None
        
        if labels is not None:
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(logits.view(-1, self.config.num_labels), labels.view(-1))
         
        if output_attentions:    
            outputs = {"loss": loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss": loss, "logits": logits}
        
        return outputs
    
class KoSpecialEntityBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config

        self.subj = tokenizer.convert_tokens_to_ids('[SUBJ]')
        self.obj = tokenizer.convert_tokens_to_ids('[OBJ]')
        
        self.pool_special_linear_block = torch.nn.Sequential(
            torch.nn.Linear(3 * self.model.config.hidden_size, self.model.config.hidden_size), 
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(self.model.config.hidden_size, config.num_labels))
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None):
        batch_size = len(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, 
            token_type_ids=token_type_ids, output_hidden_states=True)

        before_output = []
        
        for i in range(batch_size):
            each_sentence = []
            each_sentence.append(output.last_hidden_state[i][0])

            subj_idx = int((input_ids[i] == self.subj).nonzero())
            each_sentence.append(output.last_hidden_state[i][subj_idx]) 

            obj_idx = int((input_ids[i] == self.obj).nonzero())
            each_sentence.append(output.last_hidden_state[i][obj_idx])

            before_input = torch.cat(each_sentence)
            after_classifier = self.pool_special_linear_block(before_input)
            before_output.append(after_classifier)

        before_output = torch.stack(before_output)
        outputs = {'logits': before_output}

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(before_output.view(-1, self.config.num_labels), labels.view(-1))
            outputs = {'loss': loss, 'logits': before_output}

        return outputs

class CLSSpecialEntityBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config

        self.subj = tokenizer.convert_tokens_to_ids('[SUBJ]')
        self.obj = tokenizer.convert_tokens_to_ids('[OBJ]')
        
        self.pool_special_linear_block = torch.nn.Sequential(
            torch.nn.Linear(5 * self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(self.model.config.hidden_size, config.num_labels))
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None):
        batch_size = len(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, 
            token_type_ids=token_type_ids, output_hidden_states=True)

        before_output = []
        
        for i in range(batch_size):
            each_sentence = []
            each_sentence.append(output.last_hidden_state[i][0])

            subj_idx = int((input_ids[i] == self.subj).nonzero())
            subj_type_idx = subj_idx + 1
            
            each_sentence.append(output.last_hidden_state[i][subj_idx])
            each_sentence.append(output.last_hidden_state[i][subj_type_idx])

            obj_idx = int((input_ids[i] == self.obj).nonzero())
            obj_type_idx = obj_idx + 1

            each_sentence.append(output.last_hidden_state[i][obj_idx])
            each_sentence.append(output.last_hidden_state[i][obj_type_idx])

            before_input = torch.cat(each_sentence)
            after_classifier = self.pool_special_linear_block(before_input)
            before_output.append(after_classifier)
        
        before_output = torch.stack(before_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(before_output.view(-1, self.config.num_labels), labels.view(-1))
        
        outputs = {'loss': loss, 'logits': before_output}
        return outputs

class NoCLSSpecialEntityBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer):
        super().__init__(config)

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.tokenizer = tokenizer
        self.config = config

        self.subj = tokenizer.convert_tokens_to_ids('[SUBJ]')
        self.obj = tokenizer.convert_tokens_to_ids('[OBJ]')
        
        self.pool_special_linear_block = torch.nn.Sequential(
            torch.nn.Linear(4 * self.model.config.hidden_size, 4 * self.model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4 * self.model.config.hidden_size, config.num_labels))
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None):
        batch_size = len(input_ids)
        output = self.model(input_ids = input_ids, attention_mask=attention_mask, 
        token_type_ids=token_type_ids, output_hidden_states=True)

        before_output = []
        
        for i in range(batch_size):
            each_sentence = []

            subj_idx = int((input_ids[i] == self.subj).nonzero())
            subj_type_idx = subj_idx + 1 

            each_sentence.append(output.last_hidden_state[i][subj_idx])  
            each_sentence.append(output.last_hidden_state[i][subj_type_idx])

            obj_idx = int((input_ids[i] == self.obj).nonzero())
            obj_type_idx = obj_idx + 1

            each_sentence.append(output.last_hidden_state[i][obj_idx]) 
            each_sentence.append(output.last_hidden_state[i][obj_type_idx]) 

            before_input = torch.cat(each_sentence)
            after_classifier = self.pool_special_linear_block(before_input)
            before_output.append(after_classifier)

        before_output = torch.stack(before_output)
        
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(before_output.view(-1, self.config.num_labels), labels.view(-1))
        
        outputs = {'loss': loss, 'logits': before_output}
        return outputs
    
class SequentialDoubleBERT(BertPreTrainedModel):
    def __init__(self, model_name, config, tokenizer, model_type, device):
        super().__init__(config)

        self.calculated_token_count = 4

        self.tokenizer = tokenizer
        self.dv = device
        self.model_type = model_type

        self.model_config1 = AutoConfig.from_pretrained(model_name)
        self.model_config2 = AutoConfig.from_pretrained(model_name)
        self.model_config1.num_labels = 2
        self.model_config2.num_labels = 30

        if self.model_type == 'base':
            self.plm1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config1)
            self.plm2 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config2)

        elif self.model_type == 'entity_special':
            self.plm1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config1)
            self.plm2 = AutoModel.from_pretrained(model_name, config=self.model_config2)

            self.plm1.resize_token_embeddings(len(self.tokenizer))
            self.plm2.resize_token_embeddings(len(self.tokenizer))

            sub_tokens = ['[S:PER]', '[S:ORG]', '[S:LOC]', '[/S:PER]', '[/S:ORG]', '[/S:LOC]']
            token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
            self.sub_entity_token = dict(zip(sub_tokens, token_ids))

            obj_tokens = ['[O:PER]', '[O:ORG]', '[O:POH]', '[O:LOC]', '[O:NOH]', '[O:DAT]',
                  '[/O:PER]', '[/O:ORG]', '[/O:POH]', '[/O:LOC]', '[/O:NOH]', '[/O:DAT]']
            token_ids = tokenizer.convert_tokens_to_ids(obj_tokens)
            self.obj_entity_token = dict(zip(obj_tokens, token_ids))
            
            self.net = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True),
                torch.nn.ReLU())

            self.classifier = torch.nn.Linear(in_features=config.hidden_size * 2, out_features=config.num_labels, bias=True)
            
        elif self.model_type == 'entity_punct':
            self.plm1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.model_config1)
            self.plm2 = AutoModel.from_pretrained(model_name, config=self.model_config2)

            self.plm1.resize_token_embeddings(len(self.tokenizer))
            self.plm2.resize_token_embeddings(len(self.tokenizer))

            ids = tokenizer.convert_tokens_to_ids(['@', '#'])
            self.sub_ids, self.obj_ids = ids[0], ids[1]

            self.net = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True),
                torch.nn.ReLU())

            self.classifier = torch.nn.Linear(in_features=config.hidden_size * self.calculated_token_count, out_features=config.num_labels, bias=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, subject_type=None, object_type=None, subject_words=None, object_words=None):
        batch_size = len(input_ids)

        DRP_dict = {0: "는(은) 관계가 없습니다.",
                    1: "는(은) 관계가 있습니다."}
            
        if self.model_type == 'base':
            outputs1 = self.plm1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits1 = outputs1['logits']
            logits1_cpu = logits1.detach().cpu().numpy()
            result1 = np.argmax(logits1_cpu, axis=-1)

            new_sentences = list()
            tmp_binary_labels = list()

            for i in range(batch_size):
                sentence = self.tokenizer.decode(input_ids[i])[6:].replace('[PAD]', '').rstrip() + " "
                drp_sentence = subject_words[i] + "와(과) " + object_words[i] + DRP_dict[result1[i]] + ": " + f"{object_words[i]}는 {subject_words[i]}의 {object_type[i]}이다."
                new_sentences.append(sentence+drp_sentence)

                if labels != None:
                    label = 0 if labels[i] == 0 else 1
                    tmp_binary_labels.append(label)

            new_tokenized_sentence = self.tokenizer(
                new_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True)

            new_input_ids = new_tokenized_sentence['input_ids'].view(batch_size, -1).to(self.dv)
            new_attention_mask = new_tokenized_sentence['attention_mask'].view(batch_size, -1).to(self.dv)
            new_token_type_ids = new_tokenized_sentence['token_type_ids'].view(batch_size, -1).to(self.dv)
            tmp_binary_labels = torch.tensor(tmp_binary_labels).view(batch_size, -1).to(self.dv)

            outputs2 = self.plm2(new_input_ids, attention_mask=new_attention_mask, token_type_ids=new_token_type_ids)
            outputs = outputs2['logits']

            if labels is not None:
                loss_fun1 = torch.nn.BCELoss()
                loss_fun2 = torch.nn.CrossEntropyLoss()
                
                loss1 = loss_fun1(torch.argmax(logits1.view(-1, self.model_config1.num_labels), dim=1).float(), tmp_binary_labels.view(-1).float())
                loss2 = loss_fun2(outputs.view(-1, self.model_config2.num_labels), labels.view(-1))
                loss = loss1 + loss2
                
                outputs = {'loss': loss, 'logits': outputs}
            return outputs
        
        elif self.model_type == 'entity_special':
            outputs1 = self.plm1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits1 = outputs1['logits']
            logits1_cpu = logits1.detach().cpu().numpy()
            result1 = np.argmax(logits1_cpu, axis=-1)

            new_sentences = list()
            tmp_binary_labels = list()

            for i in range(batch_size):
                sentence = self.tokenizer.decode(input_ids[i])[6:].replace('[PAD]', '').rstrip() + " "
                drp_sentence = subject_words[i] + "와(과) " + object_words[i] + DRP_dict[result1[i]] + ": " + f"{object_words[i]}는 {subject_words[i]}의 {object_type[i]}이다."
                new_sentences.append(sentence+drp_sentence)

                if labels != None:
                    label = 0 if labels[i] == 0 else 1
                    tmp_binary_labels.append(label)

            new_tokenized_sentence = self.tokenizer(
                new_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True)

            new_input_ids = new_tokenized_sentence['input_ids'].view(batch_size, -1).to(self.dv)
            new_attention_mask = new_tokenized_sentence['attention_mask'].view(batch_size, -1).to(self.dv)
            new_token_type_ids = new_tokenized_sentence['token_type_ids'].view(batch_size, -1).to(self.dv)
            tmp_binary_labels = torch.tensor(tmp_binary_labels).view(batch_size, -1).to(self.dv)

            outputs2 = self.plm2(new_input_ids, attention_mask=new_attention_mask, token_type_ids=new_token_type_ids)
            hidden_states = outputs2.last_hidden_state
            batch_size, seq_length, hidden_size = hidden_states.shape

            special_hs_list = list()
 
            for i in range(batch_size):
                sub_start, sub_end = self.sub_entity_token[f'[S:{subject_type[i]}]'], self.sub_entity_token[f'[/S:{subject_type[i]}]']
                obj_start, obj_end = self.obj_entity_token[f'[O:{object_type[i]}]'], self.obj_entity_token[f'[/O:{object_type[i]}]']

                sub_start_cond = (new_input_ids[i] == sub_start).bool()
                obj_start_cond = (new_input_ids[i] == obj_start).bool()
                sub_end_cond = (new_input_ids[i] == sub_end).bool()
                obj_end_cond = (new_input_ids[i] == obj_end).bool()

                special_hs_list.append(hidden_states[i][sub_start_cond+obj_start_cond+sub_end_cond+obj_end_cond].view(-1, hidden_size))

            special_hs = torch.stack(special_hs_list, dim=0).view(batch_size, -1, hidden_size).to(self.dv)

            logits2 = self.net(special_hs).view(batch_size, -1)
            logits2 = self.classifier(logits2)

            outputs = logits2

            if labels is not None:
                loss_fun1 = torch.nn.BCELoss()
                loss_fun2 = torch.nn.CrossEntropyLoss()
                
                loss1 = loss_fun1(torch.argmax(logits1.view(-1, self.model_config1.num_labels), dim=1).float(), tmp_binary_labels.view(-1).float())
                loss2 = loss_fun2(outputs.view(-1, self.model_config2.num_labels), labels.view(-1))
                loss = loss1 + loss2
            
                outputs = {'loss': loss, 'logits': outputs}
            return outputs
        
        elif self.model_type == 'entity_punct':
            outputs1 = self.plm1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits1 = outputs1['logits']
            logits1_cpu = logits1.detach().cpu().numpy()
            result1 = np.argmax(logits1_cpu, axis=-1)

            new_sentences = list()
            tmp_binary_labels = list()

            for i in range(batch_size):
                sentence = self.tokenizer.decode(input_ids[i])[6:].replace('[PAD]', '').rstrip() + " "
                drp_sentence = subject_words[i] + "와(과) " + object_words[i] + DRP_dict[result1[i]] + ": " + f"{object_words[i]}는 {subject_words[i]}의 {object_type[i]}이다."
                new_sentences.append(sentence+drp_sentence)

                if labels != None:
                    label = 0 if labels[i] == 0 else 1
                    tmp_binary_labels.append(label)

            new_tokenized_sentence = self.tokenizer(
                new_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True)

            new_input_ids = new_tokenized_sentence['input_ids'].view(batch_size, -1).to(self.dv)
            new_attention_mask = new_tokenized_sentence['attention_mask'].view(batch_size, -1).to(self.dv)
            new_token_type_ids = new_tokenized_sentence['token_type_ids'].view(batch_size, -1).to(self.dv)
            tmp_binary_labels = torch.tensor(tmp_binary_labels).view(batch_size, -1).to(self.dv)

            outputs2 = self.plm2(new_input_ids, attention_mask=new_attention_mask, token_type_ids=new_token_type_ids)
            hidden_states = outputs2.last_hidden_state
            batch_size, seq_length, hidden_size = hidden_states.shape

            special_hs_list = list()
 
            for i in range(batch_size):
                sub_start_cond = (new_input_ids[i] == self.sub_ids).bool()
                obj_start_cond = (new_input_ids[i] == self.obj_ids).bool()

                special_hs_list.append(hidden_states[i][sub_start_cond+obj_start_cond].view(-1, hidden_size))

            special_hs = torch.stack(special_hs_list, dim=0)

            logits2 = self.net(special_hs).view(batch_size, -1)
            logits2 = self.classifier(logits2)

            outputs = logits2

            if labels is not None:
                loss_fun1 = torch.nn.BCELoss()
                loss_fun2 = torch.nn.CrossEntropyLoss()
                
                loss1 = loss_fun1(torch.argmax(logits1.view(-1, self.model_config1.num_labels), dim=1).float(), tmp_binary_labels.view(-1).float())
                loss2 = loss_fun2(outputs.view(-1, self.model_config2.num_labels), labels.view(-1))
                loss = loss1 + loss2
            
                outputs = {'loss': loss, 'logits': outputs}
            return outputs