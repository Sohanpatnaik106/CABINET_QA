import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class GPT2ModelForMaskedLM(nn.Module):

    def __init__(self, config):
        super(GPT2ModelForMaskedLM, self).__init__()


        self.config = config
        self.model_config = GPT2Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(self.config.model.model_path)
        else:
            self.model = GPT2Model.from_pretrained(self.config.model.model_path)
            self.embed_out = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, position_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids)["last_hidden_state"])
        
        return logits



class GPT2ModelForConditionalGeneration(nn.Module):

    def __init__(self, config):
        super(GPT2ModelForConditionalGeneration, self).__init__()


        self.config = config
        self.model_config = GPT2Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(self.config.model.model_path)
        else:
            self.model = GPT2Model.from_pretrained(self.config.model.model_path)
            self.embed_out = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, position_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids)["last_hidden_state"])
        
        return logits


class GPT2ModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(GPT2ModelForGenerativeQuestionAnswering, self).__init__()


        self.config = config
        self.model_config = GPT2Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(self.config.model.model_path)
        else:
            self.model = GPT2Model.from_pretrained(self.config.model.model_path)
            self.embed_out = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)

    def _get_target_modules(self):

        target_modules = [x for x in list(set([name.split(".")[-2] for name, param in self.model.named_parameters()])) if "norm" not in x]
        return target_modules


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, position_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids)["last_hidden_state"])
        
        return logits
