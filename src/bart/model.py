
import torch
import torch.nn as nn
from transformers import BartModel, BartConfig
from transformers import BartForConditionalGeneration, BartForSequenceClassification


from .modeling_bart import (
        EEDBartForConditionalGeneration, 
        RowColEEDBartForConditionalGeneration, 
        EEDBartForSequenceClassification,
        CluBartForConditionalGeneration,
        CluBartForSequenceClassification,
        ReasoningBartForConditionalGeneration,
        CluReasoningBartForConditionalGeneration,
        HighlightedCluBartForConditionalGeneration
    )

# We are training MLM as a generative task, generate the sequence of masked tokens
class BartModelForMaskedLM(nn.Module):

    def __init__(self, config):
        super(BartModelForMaskedLM, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = BartForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        return logits


# Standard generative training
class BartModelForConditionalGeneration(nn.Module):
    
    def __init__(self, config):
        super(BartModelForConditionalGeneration, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = BartForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits
    

# Classification head
class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# Sequence classification
class BartModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(BartModelForSequenceClassification, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = BartForSequenceClassification.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
        else:
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.classification_head = BartClassificationHead(input_dim = self.model.decoder.layernorm_embedding.normalized_shape[0], 
                                                          inner_dim = self.model.decoder.layernorm_embedding.normalized_shape[0],
                                                          num_classes = self.config.data.num_classes,
                                                          pooler_dropout = 0)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits
        else:
            logits = self.classification_head(self.model(input_ids = input_ids, attention_mask = attention_mask)["last_hidden_state"][:, -1, :])
        return logits


class BartModelForGenerativeQuestionAnswering(nn.Module):
    
    def __init__(self, config):
        super(BartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = BartForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits



class EEDBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(EEDBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = EEDBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs



class EEDBartModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(EEDBartModelForSequenceClassification, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = EEDBartForSequenceClassification.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())
        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.classification_head = BartClassificationHead(input_dim = self.model.decoder.layernorm_embedding.normalized_shape[0], 
                                                          inner_dim = self.model.decoder.layernorm_embedding.normalized_shape[0],
                                                          num_classes = self.config.data.num_classes,
                                                          pooler_dropout = 0)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        else:
            logits = self.classification_head(self.model(input_ids = input_ids, attention_mask = attention_mask)["last_hidden_state"][:, -1, :])
        # return logits
        return outputs






class RowColEEDBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(RowColEEDBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = RowColEEDBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None, row_ids: torch.LongTensor = None, col_ids: torch.LongTensor = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, row_ids = row_ids, col_ids = col_ids)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs


class CluBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(CluBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = CluBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs


class CluBartModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super(CluBartModelForSequenceClassification, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = CluBartForSequenceClassification.from_pretrained(self.config.model.model_path, num_labels = self.config.data.num_classes)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())
        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.classification_head = BartClassificationHead(input_dim = self.model.decoder.layernorm_embedding.normalized_shape[0], 
                                                          inner_dim = self.model.decoder.layernorm_embedding.normalized_shape[0],
                                                          num_classes = self.config.data.num_classes,
                                                          pooler_dropout = 0)

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None):

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
        else:
            logits = self.classification_head(self.model(input_ids = input_ids, attention_mask = attention_mask)["last_hidden_state"][:, -1, :])
        # return logits
        return outputs


class BartModelForTableReasoning(nn.Module):
    
    def __init__(self, config):
        super(BartModelForTableReasoning, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = BartForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits



class ReasoningBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(ReasoningBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = ReasoningBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.reason_decoder.load_state_dict(self.model.model.decoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None, 
                reason_decoder_input_ids = None, reason_labels = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, 
                                 reason_decoder_input_ids = reason_decoder_input_ids, reason_labels = reason_labels)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs




class CluReasoningBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(CluReasoningBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = CluReasoningBartForConditionalGeneration.from_pretrained(self.config.model.model_path)

            reasoning_model = BartModelForTableReasoning(self.config)
            reasoning_model.load_state_dict(torch.load("/dev/shm/tabllm/logs/table_question_reasoning_tapex_bootstrapping_baseline_loss_calc_change/checkpoints/epoch=30.pt", map_location="cpu"))

            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())
            # self.model.reasoning_model.model.decoder.load_state_dict(self.model.model.decoder.state_dict())
            self.model.reasoning_model.model.decoder.load_state_dict(reasoning_model.model.model.decoder.state_dict())
            self.model.reasoning_model.model.encoder.load_state_dict(reasoning_model.model.model.encoder.state_dict())
            self.model.reasoning_model.lm_head.load_state_dict(reasoning_model.model.lm_head.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None,
                reason_decoder_input_ids = None, reason_labels = None):
        

        if self.config.model.use_pretrained:
            qa_attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], reason_decoder_input_ids.shape[1])).to(attention_mask.device)], dim = -1)
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, 
                                 reason_decoder_input_ids = reason_decoder_input_ids, reason_labels = reason_labels,
                                 qa_attention_mask = qa_attention_mask)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs


class HighlightedCluBartModelForGenerativeQuestionAnswering(nn.Module):

    def __init__(self, config):
        super(HighlightedCluBartModelForGenerativeQuestionAnswering, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = HighlightedCluBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, 
                decoder_input_ids: torch.LongTensor = None, highlighted_cells: torch.LongTensor = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, 
                                 decoder_input_ids = decoder_input_ids, highlighted_cells = highlighted_cells)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs
    

class BartModelForLogicalFormGeneration(nn.Module):
    
    def __init__(self, config):
        super(BartModelForLogicalFormGeneration, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = BartForConditionalGeneration.from_pretrained(self.config.model.model_path)
        else:
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, decoder_input_ids: torch.LongTensor = None):
        
        if self.config.model.use_pretrained:
            logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return logits
    

class HighlightedCluBartModelForLogicalFormGeneration(nn.Module):

    def __init__(self, config):
        super(HighlightedCluBartModelForLogicalFormGeneration, self).__init__()
        
        self.config = config
        self.model_config = BartConfig.from_pretrained(self.config.model.model_path)

        if self.config.model.use_pretrained:
            self.model = HighlightedCluBartForConditionalGeneration.from_pretrained(self.config.model.model_path)
            self.model.model.decomposer.load_state_dict(self.model.model.encoder.state_dict())

        else:
            raise NotImplementedError
            self.model = BartModel.from_pretrained(self.config.model.model_path)
            self.lm_head = nn.Linear(self.model.decoder.layernorm_embedding.normalized_shape[0], self.model_config.vocab_size)


    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.LongTensor = None, 
                decoder_input_ids: torch.LongTensor = None, highlighted_cells: torch.LongTensor = None):
        

        if self.config.model.use_pretrained:
            # logits = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids).logits
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, 
                                 decoder_input_ids = decoder_input_ids, highlighted_cells = highlighted_cells)
        else:
            logits = self.lm_head(self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids)["last_hidden_state"])
        
        return outputs