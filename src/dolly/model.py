import torch
import torch.nn as nn
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, GPTNeoModel

class DollyModelForConditionalGeneration(nn.Module):

    def __init__(self, config):
        super(DollyModelForConditionalGeneration, self).__init__()


        self.config = config
        self.model_config = GPTNeoXConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = GPTNeoXForCausalLM.from_pretrained(self.config.model.model_path)
        else:
            self.model = GPTNeoModel.from_pretrained(self.config.model.model_path)
            self.embed_out = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None, position_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask)["last_hidden_state"])
        
        return logits
