from transformers import AutoModel
import torch

class Encoder_model(torch.nn.Module):
    def __init__(self, model_name,tokenizer):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_name).get_encoder()
        self.model.resize_token_embeddings(len(tokenizer))
        self.linear = torch.nn.Linear(768,1, dtype=torch.float)
        self.loss_func = torch.nn.MSELoss()
        
    def forward(self, sentence, labels=None):
        output = self.model(sentence)['last_hidden_state']
        real_output = self.linear(output)[:,0].squeeze() # cls token 이용
        
        if labels is not None:
            loss = self.loss_func(real_output, labels)
            output = {"loss" : loss, "logits" : real_output}
            
        else :
            output = {"logits" : real_output}
        
        return output
