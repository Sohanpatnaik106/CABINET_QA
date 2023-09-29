
import torch
import torch.nn as nn
from transformers import T5Model, T5Config
from transformers import T5ForConditionalGeneration

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


# Standard generative training
class T5ModelForConditionalGeneration(nn.Module):
    
    def __init__(self, config):
        super(T5ModelForConditionalGeneration, self).__init__()
        
        self.config = config
        self.model_config = T5Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = T5Model.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits


class T5ModelGenerativeQuestionAnswering(nn.Module):
    
    def __init__(self, config):
        super(T5ModelGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = T5Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = T5Model.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits

class T5ModelForTableReasoning(nn.Module):
    
    def __init__(self, config):
        super(T5ModelForTableReasoning, self).__init__()
        
        self.config = config
        self.model_config = T5Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = T5Model.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits


class T5ModelForTableCellHighlighting(nn.Module):
    
    def __init__(self, config):
        super(T5ModelForTableCellHighlighting, self).__init__()
        
        self.config = config
        self.model_config = T5Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = T5Model.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits