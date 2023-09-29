import torch
import torch.nn as nn
from .modeling_mpt import MPTForCausalLM, MPTForSequenceClassification

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


class MPTModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(MPTModelForSequenceClassification, self).__init__()

        self.config = config

        if self.config.model.use_pretrained:
            self.model = MPTForSequenceClassification.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
        else:
            raise NotImplementedError

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)

    def _get_target_modules(self):

        target_modules = [x for x in list(set([name.split(".")[-2] for name, param in self.model.named_parameters()])) if "norm" not in x and "wte" not in x and "wpe" not in x]
        return target_modules

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits
        else:
            raise NotImplementedError

        return logits


class MPTModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(MPTModelForGenerativeQuestionAnswering, self).__init__()

        self.config = config

        if self.config.model.use_pretrained:
            self.model = MPTForCausalLM.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
        else:
            raise NotImplementedError

        if self.config.model.peft:
            peft_config = LoraConfig(
                                task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, 
                                lora_dropout=0.1, target_modules = self._get_target_modules()
                            )

            self.model = get_peft_model(self.model, peft_config)

    def _get_target_modules(self):

        target_modules = [x for x in list(set([name.split(".")[-2] for name, param in self.model.named_parameters()])) if "norm" not in x and "wte" not in x and "wpe" not in x]
        return target_modules

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, position_ids: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits
        else:
            raise NotImplementedError

        return logits