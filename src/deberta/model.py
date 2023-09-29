
import torch
import torch.nn as nn
from transformers import DebertaV2ForSequenceClassification, DebertaV2Config

from .modeling_deberta import CluDebertaV2ForSequenceClassification

# Sequence classification
class DebertaModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(DebertaModelForSequenceClassification, self).__init__()
        
        self.config = config
        self.model_config = DebertaV2Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = DebertaV2ForSequenceClassification.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.classification_head = BartClassificationHead(input_dim = self.model.decoder.layernorm_embedding.normalized_shape[0], 
                                                          inner_dim = self.model.decoder.layernorm_embedding.normalized_shape[0],
                                                          num_classes = self.config.data.num_classes,
                                                          pooler_dropout = 0)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, token_type_ids: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).logits
        else:
            logits = self.classification_head(self.model(input_ids = input_ids, attention_mask = attention_mask)["last_hidden_state"][:, -1, :])
        return logits


class CluDebertaModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(CluDebertaModelForSequenceClassification, self).__init__()
        
        self.config = config
        self.model_config = DebertaV2Config.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = CluDebertaV2ForSequenceClassification.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
            self.model.deberta.decomposer.load_state_dict(self.model.deberta.encoder.state_dict())
        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.classification_head = BartClassificationHead(input_dim = self.model.decoder.layernorm_embedding.normalized_shape[0], 
                                                          inner_dim = self.model.decoder.layernorm_embedding.normalized_shape[0],
                                                          num_classes = self.config.data.num_classes,
                                                          pooler_dropout = 0)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, token_type_ids: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        else:
            logits = self.classification_head(self.model(input_ids = input_ids, attention_mask = attention_mask)["last_hidden_state"][:, -1, :])
        return outputs