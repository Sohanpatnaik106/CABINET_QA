
import os
import re
import json
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingLR

import datetime

from torchmetrics.functional import bleu_score
from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.functional.text.rouge import rouge_score
from sklearn.metrics import precision_score, recall_score, f1_score


from transformers import AdamW


from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from datasets import load_metric



from .helper import (
        to_value_list, 
        check_denotation
    )


def is_number(token):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    return bool(re.match(pattern, token))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.sum += value
        self.count += n
        self.avg = self.sum / self.count


class LightningTrainer(pl.LightningModule):

    def __init__(self, config, tokenizer, model, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super(LightningTrainer, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.model = model

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        if self.config.training.training_type == "masked_language_modelling":
            self.metrics = {"top1_accuracy": AverageMeter(),
                            "top5_accuracy": AverageMeter(),
                            "top1_numeric_accuracy": AverageMeter(),
                            "top5_numeric_accuracy": AverageMeter(),
                            "top1_text_accuracy": AverageMeter(),
                            "top5_text_accuracy": AverageMeter()}
        elif self.config.training.training_type == "description_generation":
            self.metrics = {}

        elif self.config.training.training_type == "sequence_classification":
            self.metrics = {"accuracy": AverageMeter(),
                            "precision": AverageMeter(),
                            "recall": AverageMeter(),
                            "f1": AverageMeter()}

        elif self.config.training.training_type == "column_reasoning":
            self.metrics = {"operation_exact_match": AverageMeter(), 
                           "column_exact_match": AverageMeter()}


        elif self.config.training.training_type == "table_question_answering":
            self.metrics = {"accuracy": AverageMeter()}

        elif self.config.training.training_type == "table_decomposition":
            self.metrics = {"accuracy": AverageMeter()}
        

        elif self.config.training.training_type == "descriptive_table_question_answering":
            self.metrics = {}
            metric_names = ["meteor", "bleu", "sacrebleu", "bertscore"]
            self.metric_functions = {}
            for metric in metric_names:
                self.metric_functions[metric] = load_metric(metric)

        elif self.config.training.training_type == "table_question_answering_with_reason":
            self.metrics = {"accuracy": AverageMeter()}

        elif self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
            self.metrics = {"accuracy": AverageMeter()}

        elif self.config.training.training_type == "table_question_reasoning_and_answering":
            self.metrics = {"accuracy": AverageMeter()}

        elif self.config.training.training_type == "table_cell_highlighting":
            self.metrics = {"accuracy": AverageMeter()}

        elif self.config.training.training_type == "table_logic_generation":
            self.metrics = {"accuracy": AverageMeter()}

        if self.config.training.criterion == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()


    def forward(self, batch):
        
        if self.config.model.type == "encoder-decoder":
            if self.config.training.training_type != "sequence_classification":

                if self.config.training.training_type == "table_question_answering_with_reason" or self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
                    input_ids, attention_mask, token_type_ids, decoder_input_ids, labels, reason_decoder_input_ids, reason_labels = batch
                    logits = self.model(input_ids = input_ids.to(self.device),
                                        attention_mask = attention_mask.to(self.device),
                                        decoder_input_ids = decoder_input_ids.to(self.device),
                                        reason_decoder_input_ids = reason_decoder_input_ids.to(self.device),
                                        reason_labels = reason_labels.to(self.device))

                else:
                    if self.config.tokenizer.use_row_col_ids:
                        # print("Enc-Dec Pos IDS used, No sequence classification")
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device), 
                                        decoder_input_ids = decoder_input_ids.to(self.device),
                                        row_ids = row_ids.to(self.device),
                                        col_ids = col_ids.to(self.device))
                        
                    if self.config.data.use_highlighted_cells:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, highlighted_cells, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device), 
                                        decoder_input_ids = decoder_input_ids.to(self.device),
                                        highlighted_cells = highlighted_cells.to(self.device))
                    else:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device), 
                                        decoder_input_ids = decoder_input_ids.to(self.device))

            else:

                if self.config.model.use_position_ids:
                    # print("Enc-Dec No Pos IDS used, No sequence classification")
                    input_ids, attention_mask, token_type_ids, position_ids, labels = batch
                    logits = self.model(input_ids = input_ids.to(self.device), 
                                    attention_mask = attention_mask.to(self.device),
                                    position_ids = position_ids.to(self.device))
                else:
                    input_ids, attention_mask, token_type_ids, labels = batch

                    if self.config.model.use_token_type_ids:
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device),
                                        token_type_ids = token_type_ids.to(self.device))
                    else:
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device))

        elif self.config.model.type == "decoder-only":
            if self.config.training.training_type != "sequence_classification":
        
                if self.config.training.training_type == "masked_language_modelling":
                    if self.config.model.use_position_ids:
                        # print("MLM, Position yes")
                        input_ids, attention_mask, token_type_ids, position_ids, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device),
                                        position_ids = position_ids.to(self.device))
                    else:
                        input_ids, attention_mask, token_type_ids, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device))
                    
                else:
                    if self.config.model.use_position_ids:
                        # print("Non MLM, Position yes")
                        input_ids, attention_mask, token_type_ids, position_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device),
                                        position_ids = position_ids.to(self.device))
                    else:
                        input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                        logits = self.model(input_ids = input_ids.to(self.device), 
                                        attention_mask = attention_mask.to(self.device))

            else:

                if self.config.model.use_position_ids:
                    # print("Non Position ids, sequence classification")
                    input_ids, attention_mask, token_type_ids, position_ids, labels = batch
                    logits = self.model(input_ids = input_ids.to(self.device), 
                                    attention_mask = attention_mask.to(self.device),
                                    position_ids = position_ids.to(self.device))
                else:
                    input_ids, attention_mask, token_type_ids, labels = batch
                    logits = self.model(input_ids = input_ids.to(self.device), 
                                    attention_mask = attention_mask.to(self.device))

        else:

            input_ids, attention_mask, token_type_ids, labels = batch

            if self.config.model.use_token_type_ids:
                logits = self.model(input_ids = input_ids.to(self.device), 
                                attention_mask = attention_mask.to(self.device),
                                token_type_ids = token_type_ids.to(self.device))
            else:
                logits = self.model(input_ids = input_ids.to(self.device), 
                                attention_mask = attention_mask.to(self.device))


        return logits
    

    def on_train_epoch_end(self):

        if (self.current_epoch + 1) % 5 == 0 or ((self.current_epoch + 1) == self.config.training.epochs):
            torch.save(self.model.state_dict(), os.path.join(self.config.logging.checkpoint_dir, f"epoch={self.current_epoch+1}.pt"))
        
        # NOTE: Uncomment these lines when metric computation is required for train dataloader
        # if (self.current_epoch + 1) % 5 == 0:
        # self._compute_metrics(self.train_dataloader)

        # for key, val in self.metrics.items():
        #     self.log(f"train/{key}", val.avg, sync_dist = True)
        #     print(f"train/{key}: {val.avg}", end = "\t")
        #     self.metrics[key].reset()

        # print("\n\n")

    def on_validation_epoch_end(self):
        
        # for key, val in self.metrics.items():
        #     self.log(f"test/{key}", val.avg, sync_dist = True)
        #     print(f"test/{key}: {val.avg}", end = "\t")
        #     self.metrics[key].reset()

        # if (self.current_epoch + 1) % 5 != 0:
        #     return 

        print("\n\n")

        for key, val in self.metrics.items():
            self.log(f"test/{key}", val.avg, sync_dist = True)
            print(f"test/{key}: {val.avg}", end = "\t")
            # print(f"test/{key}_correct: {val.sum}", end = "\t")
            # print(f"test/{key}_total: {val.count}", end = "\t")
            self.metrics[key].reset()

        print("\n\n")

        # NOTE: Uncomment these lines when metric computation is required for val dataloader    
        # if (self.current_epoch + 1) % 5 == 0:
        # self._compute_metrics(self.validation_dataloader)

        # for key, val in self.metrics.items():
        #     self.log(f"validation/{key}", val.avg, sync_dist = True)
        #     print(f"validation/{key}: {val.avg}", end = "\t")
        #     self.metrics[key].reset()

        # print("\n\n")

    def on_test_epoch_end(self):

        self._compute_metrics(self.test_dataloader)

        for key, val in self.metrics.items():
            self.log(f"test/{key}", val.avg, sync_dist = True)
            print(f"test/{key}: {val.avg}", end = "\t")
            self.metrics[key].reset()

        print("\n\n")

    def training_step(self, batch, batch_idx):

        # print(batch)
        outputs = self(batch)
        if self.config.model.num_encoders == 2:
            logits, reg_loss = outputs.logits, outputs.loss

        elif self.config.model.num_encoders == 3:
            logits, (reasoning_loss, reg_loss) = outputs.logits, outputs.loss
        
        elif self.config.training.training_type == "table_question_answering_with_reason":
            logits, reasoning_loss = outputs.logits, outputs.loss

        else:
            logits = outputs
        # logits = outputs.logits
        # logits = outputs

        if self.config.training.training_type == "table_question_answering_with_reason":
            loss = self._compute_loss(logits, batch[-3])
        elif self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
            loss = self._compute_loss(logits, batch[-3])
        else:    
            loss = self._compute_loss(logits, batch[-1])

        self.log("train/loss", loss.item(), prog_bar = True, sync_dist = True)

        if self.config.model.num_encoders == 2:
            self.log("train/reg_loss", reg_loss.item(), prog_bar=True, sync_dist=True)

        elif self.config.model.num_encoders == 3:
            self.log("train/reasoning_loss", reasoning_loss.item(), prog_bar=True, sync_dist=True)
            self.log("train/reg_loss", reg_loss.item(), prog_bar=True, sync_dist=True)
        
        if self.config.training.training_type == "table_question_answering_with_reason":
            self.log("train/reasoning_loss", reasoning_loss.item(), prog_bar=True, sync_dist=True)

        total_loss = loss
        if self.config.model.num_encoders == 2:
            total_loss += 0.1 * reg_loss

        elif self.config.model.num_encoders == 3:
            total_loss += (reasoning_loss + 0.1 * reg_loss)

        if self.config.training.training_type == "table_question_answering_with_reason":
            total_loss += reasoning_loss

        return total_loss
    

    def validation_step(self, batch, batch_idx):
        
        outputs = self(batch)
        if self.config.model.num_encoders == 2:
            logits, reg_loss = outputs.logits, outputs.loss

        elif self.config.model.num_encoders == 3:
            logits, (reasoning_loss, reg_loss) = outputs.logits, outputs.loss

        elif self.config.training.training_type == "table_question_answering_with_reason":
            logits, reasoning_loss = outputs.logits, outputs.loss

        else:
            logits = outputs
        # logits = outputs.logits
        # logits = outputs

        if self.config.training.training_type == "table_question_answering_with_reason":
            loss = self._compute_loss(logits, batch[-3])
        elif self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
            loss = self._compute_loss(logits, batch[-3])
        else:    
            loss = self._compute_loss(logits, batch[-1])
        self.log("validation/loss", loss.item(), prog_bar = True, sync_dist = True)

        if self.config.model.num_encoders == 2:
            self.log("validation/reg_loss", reg_loss.item(), prog_bar=True, sync_dist=True)

        elif self.config.model.num_encoders == 3:
            self.log("validation/reasoning_loss", reasoning_loss.item(), prog_bar=True, sync_dist=True)
            self.log("validation/reg_loss", reg_loss.item(), prog_bar=True, sync_dist=True)

        if self.config.training.training_type == "table_question_answering_with_reason":
            self.log("validation/reasoning_loss", reasoning_loss.item(), prog_bar=True, sync_dist=True)

        # if (self.current_epoch + 1) % 5 != 0:
        #     return loss + reg_loss

        if self.config.training.training_type == "masked_language_modelling":
  
            logits = self(batch)
            labels = batch[-1]

            for logit, label in zip(logits, labels):
                masked_indices = (label != -100)
                logit = logit[:masked_indices.shape[0], :]

                # Flatten the logits and output_ids tensors for masked tokens
                masked_logits = logit[masked_indices == True, :]
                masked_output_ids = label[masked_indices == True]

                masked_logits = masked_logits.cpu().detach()
                masked_output_ids = masked_output_ids.cpu().detach()


                # Calculate top-1 accuracy for masked tokens
                self.metrics["top1_accuracy"].update(torch.sum(torch.argmax(masked_logits, dim=1) == masked_output_ids).float(), n = masked_logits.shape[0])

                # Calculate top-5 accuracy for masked tokens
                _, top5_predictions = torch.topk(masked_logits, k=5, dim=1)

                self.metrics["top5_accuracy"].update(torch.sum(torch.any(top5_predictions == masked_output_ids.view(-1, 1), dim=1).float()), n = masked_logits.shape[0])

                # Convert logits and output_ids to numpy arrays for masked tokens
                
                numeric_output_idx = []
                text_output_idx = []
                for i, output_id in enumerate(masked_output_ids):
                    token = self.tokenizer.convert_ids_to_tokens(torch.tensor([output_id]))[0].replace("Ġ", "")
                    if is_number(token):
                        numeric_output_idx.append(i)
                    else:
                        text_output_idx.append(i)

                numeric_output_idx = torch.tensor(numeric_output_idx)

                if len(numeric_output_idx) != 0:
                    
                    numeric_output_ids = torch.tensor(masked_output_ids[numeric_output_idx])
                    numeric_logits = torch.tensor(masked_logits[numeric_output_idx])

                    if len(numeric_logits.shape) == 1:
                        numeric_logits = numeric_logits.unsqueeze(0)

                    self.metrics["top1_numeric_accuracy"].update(torch.sum((torch.argmax(numeric_logits, dim=1) == numeric_output_ids).float()), n = numeric_logits.shape[0])

                    # Calculate top-5 accuracy for masked tokens
                    _, numeric_top5_predictions = torch.topk(numeric_logits, k=5, dim=1)
                    self.metrics["top5_numeric_accuracy"].update(torch.sum(torch.any(numeric_top5_predictions == numeric_output_ids.view(-1, 1), dim=1).float()), n = numeric_logits.shape[0])

                text_output_idx = torch.tensor(text_output_idx)
                if len(text_output_idx) != 0:
                    text_output_ids = torch.tensor(masked_output_ids[text_output_idx])
                    text_logits = torch.tensor(masked_logits[text_output_idx])

                    if len(text_logits.shape) == 1:
                        text_logits = text_logits.unsqueeze(0)

                    self.metrics["top1_text_accuracy"].update(torch.sum((torch.argmax(text_logits, dim=1) == text_output_ids).float()), n = text_logits.shape[0])

                    # Calculate top-5 accuracy for masked tokens
                    _, text_top5_predictions = torch.topk(text_logits, k=5, dim=1)

                    self.metrics["top5_text_accuracy"].update(torch.sum(torch.any(text_top5_predictions == text_output_ids.view(-1, 1), dim=1).float()), n = text_logits.shape[0])

        
        elif self.config.training.training_type == "description_generation":

                    
            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":
                    input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                    actual_output_ids = labels.clone()
                    output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            metrics = {
                # TODO: See why for bert score the checkpoint loading warning is coming
                # "bert_score": {key: np.mean(val) for key, val in bert_score(preds, target, model_name_or_path = config.metrics.bert_score_model).items()},
                "bleu_score": bleu_score(predicted_sequence, actual_sequence),
                # "perplexity": perplexity(logits, labels, ignore_index = self.config.data.pad_token_id),
                "rouge_score": rouge_score(predicted_sequence, actual_sequence)
            }

            if len(list(self.metrics.keys())) == 0:
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            self.metrics[f"{key}_{k}"] = AverageMeter()
                            self.metrics[f"{key}_{k}"].update(v * input_ids.shape[0], input_ids.shape[0])
                    else:
                        self.metrics[f"{key}"] = AverageMeter()
                        self.metrics[f"{key}"].update(val * input_ids.shape[0], input_ids.shape[0])

            else:
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            self.metrics[f"{key}_{k}"].update(v * input_ids.shape[0], input_ids.shape[0])
                    else:
                        self.metrics[f"{key}"].update(val * input_ids.shape[0], input_ids.shape[0])


        elif self.config.training.training_type == "sequence_classification":

            labels = batch[-1].cpu()

            # logits = self(batch).detach().cpu()
            predicted_labels = torch.argmax(logits, dim = -1)

            self.metrics["accuracy"].update(torch.sum((predicted_labels.cpu() == labels).float()), labels.shape[0])
            self.metrics["precision"].update(precision_score(predicted_labels.cpu().numpy(), labels.numpy()) * labels.shape[0], labels.shape[0])
            self.metrics["recall"].update(recall_score(predicted_labels.cpu().numpy(), labels.numpy()) * labels.shape[0], labels.shape[0])
            self.metrics["f1"].update(f1_score(predicted_labels.cpu().numpy(), labels.numpy()) * labels.shape[0], labels.shape[0])


        elif self.config.training.training_type == "column_reasoning":

                    
            if self.config.model.use_position_ids:
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":
                    input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                    actual_output_ids = labels.clone()
                    output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))


            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")

            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            for a, p in zip(actual_sequence, predicted_sequence):
                if "greater than mean" in a:
                    if "greater than mean" in p:
                        self.metrics["operation_exact_match"].update(1, 1)
                        col = a.split("value of")[1].split("is")[0]
                        if col in p:
                            self.metrics["column_exact_match"].update(1, 1) 
                    else:
                        self.metrics["operation_exact_match"].update(0, 1)

                if "greater than median" in a:
                    if "greater than median" in p:
                        self.metrics["operation_exact_match"].update(1, 1)
                        col = a.split("value of")[1].split("is")[0]
                        if col in p:
                            self.metrics["column_exact_match"].update(1, 1) 
                    else:
                        self.metrics["operation_exact_match"].update(0, 1)

                elif "weighted combination" in a:
                    if "weighted combination" in p:
                        self.metrics["operation_exact_match"].update(1, 1)
                        correct = 0
                        total = 0
                        columns = [col.strip() for col in a.split("combination of")[1].split("and")]
                        # if len(columns) == 2:
                        for col in columns:
                            if col in p:
                                correct += 1
                            total += 1
                        self.metrics["column_exact_match"].update(correct, total)
                    else:
                        self.metrics["operation_exact_match"].update(0, 1)

                elif "multiplying" in a:
                    if "multiplying" in p:
                        self.metrics["operation_exact_match"].update(1, 1)
                        columns = [col.strip() for col in a.split("multiplying")[1].split("and")]
                        correct = 0
                        total = 0
                        for col in columns:
                            if col in p:
                                correct += 1
                            total += 1
                        self.metrics["column_exact_match"].update(correct, total)
                    else:
                        self.metrics["operation_exact_match"].update(0, 1)

                else:
                    if "correlation" in p:
                        self.metrics["operation_exact_match"].update(1, 1)
                    else:
                        self.metrics["operation_exact_match"].update(0, 1)

        elif self.config.training.training_type == "table_question_answering":

            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":

                    if self.config.tokenizer.use_row_col_ids:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                    if self.config.data.use_highlighted_cells:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, highlighted_cells, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            highlighted_cells = highlighted_cells.to(self.device))
                    else:    
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            for a, p in zip(actual_sequence, predicted_sequence):
                a = [x.strip() for x in a.split(",")]
                p = [x.strip() for x in p.split(",")]

                # pred = to_value_list([p])
                # gold = to_value_list([a])

                pred = to_value_list(p)
                gold = to_value_list(a)

                correct = check_denotation(gold, pred)

                if correct:
                    self.metrics["accuracy"].update(1, 1)
                else:
                    self.metrics["accuracy"].update(0, 1)

        elif self.config.training.training_type == "table_decomposition":

                    
            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":
                    input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                    actual_output_ids = labels.clone()
                    output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            for a, p in zip(actual_sequence, predicted_sequence):

                gold = a.split()
                pred = p.split()

                gold_freq = Counter(gold)
                pred_freq = Counter(pred)

                correct = sum(min(gold_freq[word], pred_freq[word]) for word in pred)
                total = len(gold)

                self.metrics["accuracy"].update(correct, total)

        elif self.config.training.training_type == "descriptive_table_question_answering":
            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":

                    if self.config.tokenizer.use_row_col_ids:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                        actual_output_ids = labels.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                    if self.config.data.use_highlighted_cells:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, highlighted_cells, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            highlighted_cells = highlighted_cells.to(self.device))

                    else:    
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)
            
            metric_names = ["meteor", "bleu", "sacrebleu", "bertscore"]

            # metric = load_metric('sacrebleu')
            # res = metric.compute(predictions = predicted_sequence, references = actual_sequence)
            # self.metrics['bleu'] = res["score"]

            for metric_name in metric_names:

                # metric = load_metric(metric_name)
                metric = self.metric_functions[metric_name]
                # decoded_preds, decoded_labels = postprocess_text(decoded_raw_preds, decoded_raw_labels, metric_name)
                
                
                if metric_name == "bertscore":
                    res = metric.compute(predictions = predicted_sequence, references = actual_sequence,lang="en")
                    for k,v in res.items():
                        if k == "hashcode":
                            continue
                        
                        if f"{metric_name}_{k}_0" not in list(self.metrics.keys()):
                            self.metrics[f"{metric_name}_{k}_0"] = AverageMeter()

                        # if f"{metric_name}_{k}_1" not in list(self.metrics.keys()):
                        #     self.metrics[f"{metric_name}_{k}_1"] = AverageMeter()

                        self.metrics[f"{metric_name}_{k}_0"].update(v[0], 1) # round(v[0], 2)
                        # self.metrics[f"{metric_name}_{k}_1"].update(v[1], 1) # round(v[1], 2)

                else:
                    if "bleu" in metric_name:
                        actual_sequence = [[actual_sequence]]
                        predicted_sequence = [predicted_sequence]
                        
                    res = metric.compute(predictions = predicted_sequence, references = actual_sequence)
                    
                    if metric_name == "sacrebleu":
                        if f"{metric_name}" not in list(self.metrics.keys()):
                            self.metrics[metric_name] = AverageMeter()
                        self.metrics[metric_name].update(res["score"], 1)

                    elif metric_name == "bleurt":
                        if f"{metric_name}_0" not in list(self.metrics.keys()):
                            self.metrics[f"{metric_name}_0"] = AverageMeter()

                        # if f"{metric_name}_1" not in list(self.metrics.keys()):
                        #     self.metrics[f"{metric_name}_1"] = AverageMeter()

                        self.metrics[f"{metric_name}_0"].update(res["scores"][0], 1) # round(res["scores"][0], 2) 
                        # self.metrics[f"{metric_name}_1"].update(res["scores"][1], 1) # round(res["scores"][1], 2) 

                    else:
                        if metric_name not in list(self.metrics.keys()):
                            self.metrics[metric_name] = AverageMeter()

                        self.metrics[metric_name].update(res[metric_name], 1)


        elif self.config.training.training_type == "table_question_answering_with_reason":

            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":
                    
                    if self.config.training.training_type == "table_question_answering_with_reason" or self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels, reason_decoder_input_ids, reason_labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))

                    else:
                        if self.config.tokenizer.use_row_col_ids:
                            input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                            actual_output_ids = labels.clone()
                            output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                                row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                        else:    
                            input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                            actual_output_ids = labels.clone()
                            output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                    num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            for a, p in zip(actual_sequence, predicted_sequence):

                a = [x.strip() for x in a.split(",")]
                p = [x.strip() for x in p.split(",")]

                # pred = to_value_list([p])
                # gold = to_value_list([a])

                pred = to_value_list(p)
                gold = to_value_list(a)

                correct = check_denotation(gold, pred)

                if correct:
                    self.metrics["accuracy"].update(1, 1)
                else:
                    self.metrics["accuracy"].update(0, 1)

        elif self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
            

            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":
                    
                    if self.config.training.training_type == "table_question_answering_with_reason" or self.config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels, reason_decoder_input_ids, reason_labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))

                    else:
                        if self.config.tokenizer.use_row_col_ids:
                            input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                            actual_output_ids = labels.clone()
                            output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                                row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                        else:    
                            input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                            actual_output_ids = labels.clone()
                            output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                    num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            for a, p in zip(actual_sequence, predicted_sequence):

                a = [x.strip() for x in a.split(",")]
                p = [x.strip() for x in p.split(",")]

                # pred = to_value_list([p])
                # gold = to_value_list([a])

                pred = to_value_list(p)
                gold = to_value_list(a)

                correct = check_denotation(gold, pred)

                if correct:
                    self.metrics["accuracy"].update(1, 1)
                else:
                    self.metrics["accuracy"].update(0, 1)


        elif self.config.training.training_type == "table_question_reasoning_and_answering":
            
            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":

                    if self.config.tokenizer.use_row_col_ids:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                    else:    
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            predicted_sequence = [p.split("answer: ")[-1].strip() for p in predicted_sequence]
            # predicted_sequence = predicted_sequence.split("answer: ")[-1].strip()

            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)
            actual_sequence = [a.split("answer: ")[-1].strip() for a in actual_sequence]
            # actual_sequence = actual_sequence.split("answer: ")[-1].strip()


            for a, p in zip(actual_sequence, predicted_sequence):
                a = [x.strip() for x in a.split(",")]
                p = [x.strip() for x in p.split(",")]

                # pred = to_value_list([p])
                # gold = to_value_list([a])

                pred = to_value_list(p)
                gold = to_value_list(a)

                correct = check_denotation(gold, pred)

                if correct:
                    self.metrics["accuracy"].update(1, 1)
                else:
                    self.metrics["accuracy"].update(0, 1)


        elif self.config.training.training_type == "table_cell_highlighting":
            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":

                    if self.config.tokenizer.use_row_col_ids:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                    else:    
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # predicted_sequence = predicted_sequence.split("answer: ")[-1].strip()

            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)
            # actual_sequence = actual_sequence.split("answer: ")[-1].strip()


            for a, p in zip(actual_sequence, predicted_sequence):
                a = [x.strip() for x in a.split(",")]
                p = [x.strip() for x in p.split(",")]

                # pred = to_value_list([p])
                # gold = to_value_list([a])

                pred = to_value_list(p)
                gold = to_value_list(a)

                correct = check_denotation(gold, pred)

                if correct:
                    self.metrics["accuracy"].update(1, 1)
                else:
                    self.metrics["accuracy"].update(0, 1)

        elif self.config.training.training_type == "table_logic_generation":
            if self.config.model.use_position_ids:
                raise NotImplementedError
                input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

                # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
                output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
                                                        max_length = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

            else:
                if self.config.model.type == "encoder-decoder":

                    if self.config.tokenizer.use_row_col_ids:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, row_ids, col_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            row_ids = row_ids.to(self.device), col_ids = col_ids.to(self.device))

                    if self.config.data.use_highlighted_cells:
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, highlighted_cells, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device), 
                                                            highlighted_cells = highlighted_cells.to(self.device))
                    else:    
                        input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
                        actual_output_ids = decoder_input_ids.clone()
                        output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                                num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
                        
                elif self.config.model.type == "decoder-only":
                    input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
                    output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
                                                        num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

            predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if self.config.model.type == "decoder-only":
                for i in range(len(predicted_sequence)):
                    predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
            
            actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

            for a, p in zip(actual_sequence, predicted_sequence):
                a = a.strip()
                p = p.strip()

                if p == a:
                    self.metrics["accuracy"].update(1, 1)
                else:
                    self.metrics["accuracy"].update(0, 1)



        # if self.config.model.use_position_ids:
        #     raise NotImplementedError
        #     input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

        #     # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
        #     output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
        #                                             max_length = self.config.tokenizer.output_max_length, 
        #                                             num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

        # else:
        #     if self.config.model.type == "encoder-decoder":
        #         input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
        #         actual_output_ids = decoder_input_ids.clone()
        #         output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
        #                                                 num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
        #     elif self.config.model.type == "decoder-only":
        #         input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
        #         output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
        #                                             num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

        # predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # if self.config.model.type == "decoder-only":
        #     for i in range(len(predicted_sequence)):
        #         predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
        
        # actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

        # for a, p in zip(actual_sequence, predicted_sequence):
        #     pred = to_value_list([p])
        #     gold = to_value_list([a])
        #     correct = check_denotation(pred, gold)

        #     if correct:
        #         self.metrics["accuracy"].update(1, 1)
        #     else:
        #         self.metrics["accuracy"].update(0, 1)

        # d = {"actual": actual_sequence, "pred": predicted_sequence}
        # # Get the current time
        # current_time = datetime.datetime.now()

        # # Convert it to a string
        # time_string = current_time.strftime("%H:%M:%S")

        # with open(os.path.join(self.config.logging.log_dir, f"{time_string}_predictions.json"), "w") as f:
        #     json.dump(d, f)
        

        total_loss = loss
        if self.config.model.num_encoders == 2:
            total_loss += 0.1 * reg_loss

        elif self.config.model.num_encoders == 3:
            total_loss += (reasoning_loss + 0.1 * reg_loss)

        if self.config.training.training_type == "table_question_answering_with_reason":
            total_loss += reasoning_loss

        return total_loss
        # return loss

    def test_step(self, batch, batch_idx):
        pass
        outputs = self(batch)
        logits, reg_loss = outputs.logits, outputs.loss
        # logits = outputs.logits
        # logits = outputs

        loss = self._compute_loss(logits, batch[-1])
        self.log("test/loss", loss.item(), prog_bar = True, sync_dist = True)
        self.log("test/reg_loss", reg_loss.item(), prog_bar = True, sync_dist = True)

        return loss + 0.1 * reg_loss
        # return losspython
    

    # NOTE: Try out different settings of optimisation schemes

    # def get_optimizer_grouped_parameters(self, model, learning_rate, weight_decay, layerwise_learning_rate_decay):

    #     no_decay = ["bias"]
    #     # initialize lr for task specific layer
    #     optimizer_grouped_parameters = [
    #         {
    #             # "params": [p for p in model.model.lm_head.parameters()] + [p for n, p in model.named_parameters() if "shared" in n] + [p for n, p in model.named_parameters() if "token_classifier" in n],
    #             "params": [p for n, p in model.named_parameters() if "shared" in n] + [p for n, p in model.named_parameters() if "token_classifier" in n],
    #             # "params": [p for n, p in model.named_parameters() if "shared" in n] + [p for n, p in model.named_parameters() if "token_classifier" in n],
    #             "weight_decay": 0.0,
    #             "lr": learning_rate,
    #         },
    #     ]

    #     optimizer_grouped_parameters += [
    #         {
    #             "params": [p for n, p in model.model.model.decomposer.layernorm_embedding.named_parameters() if "bias" not in n] + [p for n, p in model.model.model.encoder.layernorm_embedding.named_parameters() if "bias" not in n] + [p for n, p in model.model.model.decoder.layernorm_embedding.named_parameters() if "bias" not in n],
    #             # "params": [p for n, p in model.named_parameters() if "shared" in n] + [p for n, p in model.named_parameters() if "token_classifier" in n],
    #             "weight_decay": 0.0,
    #             "lr": learning_rate,
    #         },
    #     ]

    #     optimizer_grouped_parameters += [
    #         {
    #             "params": [p for n, p in model.model.model.decomposer.named_parameters() if "bias" in n] + [p for n, p in model.model.model.encoder.named_parameters() if "bias" in n] + [p for n, p in model.model.model.decoder.named_parameters() if "bias" in n],
    #             "weight_decay": 0.0,
    #             "lr": learning_rate,
    #         },
    #     ]

    #     # decomposer_layers = [getattr(model.model.model, "decomposer").embed_tokens] + [getattr(model.model.model, "decomposer").embed_positions] \
    #     #                         + list(getattr(model.model.model, "decomposer").layers)

    #     decomposer_layers = [getattr(model.model.model, "decomposer").embed_positions] + list(getattr(model.model.model, "decomposer").layers)
    #     # decomposer_layers = list(getattr(model.model.model, "decomposer").layers)

        
    #     # encoder_layers = [getattr(model.model.model, "encoder").embed_tokens] + [getattr(model.model.model, "encoder").embed_positions] \
    #     #                         + list(getattr(model.model.model, "encoder").layers)

    #     encoder_layers = [getattr(model.model.model, "encoder").embed_positions] + list(getattr(model.model.model, "encoder").layers)
    #     # encoder_layers = list(getattr(model.model.model, "encoder").layers)

    #     # decoder_layers = [getattr(model.model.model, "decoder").embed_tokens] + [getattr(model.model.model, "decoder").embed_positions] \
    #                             # + list(getattr(model.model.model, "decoder").layers)

    #     decoder_layers = [getattr(model.model.model, "decoder").embed_positions] + list(getattr(model.model.model, "decoder").layers)
    #     # decoder_layers = list(getattr(model.model.model, "decoder").layers)

    #     # decomposer_layers = list(getattr(model.model, "model").decomposer)
    #     # encoder_layers = list(getattr(model.model, "model").encoder)
    #     # decoder_layers = list(getattr(model.model, "model").decoder)
    #     # # layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)



    #     # # layers.reverse()
    #     decomposer_layers.reverse()
    #     encoder_layers.reverse()
    #     decoder_layers.reverse()

    #     lr = learning_rate
    #     for layer in decomposer_layers:
    #         lr *= layerwise_learning_rate_decay
    #         optimizer_grouped_parameters += [
    #             {
    #                 "params": [p for n, p in layer.named_parameters() if "bias" not in n],
    #                 "weight_decay": weight_decay,
    #                 "lr": lr,
    #             },
    #             # {
    #             #     "params": [p for n, p in layer.named_parameters() if "bias" in n],
    #             #     "weight_decay": 0.0,
    #             #     "lr": lr,
    #             # },
    #         ]

    #     lr = learning_rate
    #     for layer in encoder_layers:
    #         lr *= layerwise_learning_rate_decay
    #         optimizer_grouped_parameters += [
    #             {
    #                 "params": [p for n, p in layer.named_parameters() if "bias" not in n],
    #                 "weight_decay": weight_decay,
    #                 "lr": lr,
    #             },
    #             # {
    #             #     "params": [p for n, p in layer.named_parameters() if "bias" in n],
    #             #     "weight_decay": 0.0,
    #             #     "lr": lr,
    #             # },
    #         ]

    #     lr = learning_rate
    #     for layer in decoder_layers:
    #         lr *= layerwise_learning_rate_decay
    #         optimizer_grouped_parameters += [
    #             {
    #                 "params": [p for n, p in layer.named_parameters() if "bias" not in n],
    #                 "weight_decay": weight_decay,
    #                 "lr": lr,
    #             },
    #             # {
    #             #     "params": [p for n, p in layer.named_parameters() if "bias" in n],
    #             #     "weight_decay": 0.0,
    #             #     "lr": lr,
    #             # },
    #         ]
            

    #     return optimizer_grouped_parameters


    def configure_optimizers(self):

        # learning_rate = self.config.training.lr
        # layerwise_learning_rate_decay = self.config.training.layerwise_learning_rate_decay # 0.9
        # weight_decay = self.config.training.weight_decay  # 0.01 
        # adam_epsilon = self.config.training.adam_epsilon   # 1e-6
        # use_bertadam = self.config.training.use_bertadam  # False
        # # scheduler params
        # num_epochs = self.config.training.epochs # 20
        # num_warmup_steps = self.config.training.num_warmup_steps # 0

        # # # optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.config.training.lr)

        # grouped_optimizer_params = self.get_optimizer_grouped_parameters(
        #     self.model,
        #     learning_rate, weight_decay, 
        #     layerwise_learning_rate_decay
        # )
        # optimizer = AdamW(
        #     grouped_optimizer_params,
        #     lr=learning_rate,
        #     eps=adam_epsilon,
        #     correct_bias=not use_bertadam
        # )
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=num_epochs
        # )

        # return [optimizer], [scheduler]   

        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.config.training.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)

        return [optimizer], [scheduler]

        # return torch.optim.AdamW(self.trainer.model.parameters(), lr=self.config.training.lr)



    
    
    def _compute_loss(self, logits, labels):
        
        if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" or \
            self.config.training.training_type == "masked_language_modelling" or self.config.training.training_type == "table_decomposition":
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, :labels.shape[1], :]
                logits = logits.contiguous()

        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # NOTE: The following lines are needed when all the labels are -100 in cross entropy loss
        # if labels[labels == -100].sum() == labels.shape[0]:
        #     loss = loss * 0

        # if torch.isnan(loss):
        #     loss = torch.tensor(0., requires_grad = True, device = logits.device)

        return loss











    # def _compute_metrics(self, dataloader):
        

    #     if self.config.training.training_type == "masked_language_modelling":
    #         with tqdm(dataloader, unit = "batch", position = 0, leave = True, desc = "Metrics") as tepoch:
    #             for batch_idx, batch in enumerate(tepoch):
                    
    #                 logits = self(batch)
    #                 labels = batch[-1]

    #                 for logit, label in zip(logits, labels):
    #                     masked_indices = (label != -100)
    #                     logit = logit[:masked_indices.shape[0], :]

    #                     # Flatten the logits and output_ids tensors for masked tokens
    #                     masked_logits = logit[masked_indices == True, :]
    #                     masked_output_ids = label[masked_indices == True]

    #                     masked_logits = masked_logits.cpu().detach()
    #                     masked_output_ids = masked_output_ids.cpu().detach()


    #                     # Calculate top-1 accuracy for masked tokens
    #                     self.metrics["top1_accuracy"].update(torch.sum(torch.argmax(masked_logits, dim=1) == masked_output_ids).float(), n = masked_logits.shape[0])

    #                     # Calculate top-5 accuracy for masked tokens
    #                     _, top5_predictions = torch.topk(masked_logits, k=5, dim=1)

    #                     self.metrics["top5_accuracy"].update(torch.sum(torch.any(top5_predictions == masked_output_ids.view(-1, 1), dim=1).float()), n = masked_logits.shape[0])

    #                     # Convert logits and output_ids to numpy arrays for masked tokens
                        
    #                     numeric_output_idx = []
    #                     text_output_idx = []
    #                     for i, output_id in enumerate(masked_output_ids):
    #                         token = self.tokenizer.convert_ids_to_tokens(torch.tensor([output_id]))[0].replace("Ġ", "")
    #                         if is_number(token):
    #                             numeric_output_idx.append(i)
    #                         else:
    #                             text_output_idx.append(i)

    #                     numeric_output_idx = torch.tensor(numeric_output_idx)

    #                     if len(numeric_output_idx) != 0:
                            
    #                         numeric_output_ids = torch.tensor(masked_output_ids[numeric_output_idx])
    #                         numeric_logits = torch.tensor(masked_logits[numeric_output_idx])

    #                         if len(numeric_logits.shape) == 1:
    #                             numeric_logits = numeric_logits.unsqueeze(0)

    #                         self.metrics["top1_numeric_accuracy"].update(torch.sum((torch.argmax(numeric_logits, dim=1) == numeric_output_ids).float()), n = numeric_logits.shape[0])

    #                         # Calculate top-5 accuracy for masked tokens
    #                         _, numeric_top5_predictions = torch.topk(numeric_logits, k=5, dim=1)
    #                         self.metrics["top5_numeric_accuracy"].update(torch.sum(torch.any(numeric_top5_predictions == numeric_output_ids.view(-1, 1), dim=1).float()), n = numeric_logits.shape[0])

    #                     text_output_idx = torch.tensor(text_output_idx)
    #                     if len(text_output_idx) != 0:
    #                         text_output_ids = torch.tensor(masked_output_ids[text_output_idx])
    #                         text_logits = torch.tensor(masked_logits[text_output_idx])

    #                         if len(text_logits.shape) == 1:
    #                             text_logits = text_logits.unsqueeze(0)

    #                         self.metrics["top1_text_accuracy"].update(torch.sum((torch.argmax(text_logits, dim=1) == text_output_ids).float()), n = text_logits.shape[0])

    #                         # Calculate top-5 accuracy for masked tokens
    #                         _, text_top5_predictions = torch.topk(text_logits, k=5, dim=1)

    #                         self.metrics["top5_text_accuracy"].update(torch.sum(torch.any(text_top5_predictions == text_output_ids.view(-1, 1), dim=1).float()), n = text_logits.shape[0])

        
    #     elif self.config.training.training_type == "description_generation":
    #         with tqdm(dataloader, unit = "batch", position = 0, leave = True, desc = "Metrics") as tepoch:
    #             for batch_idx, batch in enumerate(tepoch):
                    
    #                 if self.config.model.use_position_ids:
    #                     raise NotImplementedError
    #                     input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

    #                     # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
    #                     output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
    #                                                             max_length = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

    #                 else:
    #                     if self.config.model.type == "encoder-decoder":
    #                         input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
    #                         actual_output_ids = labels.clone()
    #                         output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                                 num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
    #                     elif self.config.model.type == "decoder-only":
    #                         input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
    #                         output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

    #                 predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #                 if self.config.model.type == "decoder-only":
    #                     for i in range(len(predicted_sequence)):
    #                         predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
                    
    #                 actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

    #                 metrics = {
    #                     # TODO: See why for bert score the checkpoint loading warning is coming
    #                     # "bert_score": {key: np.mean(val) for key, val in bert_score(preds, target, model_name_or_path = config.metrics.bert_score_model).items()},
    #                     "bleu_score": bleu_score(predicted_sequence, actual_sequence),
    #                     # "perplexity": perplexity(logits, labels, ignore_index = self.config.data.pad_token_id),
    #                     "rouge_score": rouge_score(predicted_sequence, actual_sequence)
    #                 }

    #                 if len(list(self.metrics.keys())) == 0:
    #                     for key, val in metrics.items():
    #                         if isinstance(val, dict):
    #                             for k, v in val.items():
    #                                 self.metrics[f"{key}_{k}"] = AverageMeter()
    #                                 self.metrics[f"{key}_{k}"].update(v * input_ids.shape[0], input_ids.shape[0])
    #                         else:
    #                             self.metrics[f"{key}"] = AverageMeter()
    #                             self.metrics[f"{key}"].update(val * input_ids.shape[0], input_ids.shape[0])

    #                 else:
    #                     for key, val in metrics.items():
    #                         if isinstance(val, dict):
    #                             for k, v in val.items():
    #                                 self.metrics[f"{key}_{k}"].update(v * input_ids.shape[0], input_ids.shape[0])
    #                         else:
    #                             self.metrics[f"{key}"].update(val * input_ids.shape[0], input_ids.shape[0])


    #     elif self.config.training.training_type == "sequence_classification":

    #         with tqdm(dataloader, unit = "batch", position = 0, leave = True, desc = "Metrics") as tepoch:
    #             for batch_idx, batch in enumerate(tepoch):
                    
    #                 labels = batch[-1]

    #                 logits = self(batch).detach().cpu()
    #                 predicted_labels = torch.argmax(logits, dim = -1)

    #                 self.metrics["accuracy"].update(torch.sum((predicted_labels == labels).float()), labels.shape[0])
    #                 self.metrics["precision"].update(precision_score(predicted_labels.numpy(), labels.numpy()) * labels.shape[0], labels.shape[0])
    #                 self.metrics["recall"].update(recall_score(predicted_labels.numpy(), labels.numpy()) * labels.shape[0], labels.shape[0])
    #                 self.metrics["f1"].update(f1_score(predicted_labels.numpy(), labels.numpy()) * labels.shape[0], labels.shape[0])


    #     elif self.config.training.training_type == "column_reasoning":
    #         with tqdm(dataloader, unit = "batch", position = 0, leave = True, desc = "Metrics") as tepoch:
    #             for batch_idx, batch in enumerate(tepoch):
                    
    #                 if self.config.model.use_position_ids:
    #                     input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch
    #                     output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
    #                                                             max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

    #                 else:
    #                     if self.config.model.type == "encoder-decoder":
    #                         input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
    #                         actual_output_ids = labels.clone()
    #                         output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
    #                     elif self.config.model.type == "decoder-only":
    #                         input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
    #                         output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))


    #                 predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    #                 if self.config.model.type == "decoder-only":
    #                     for i in range(len(predicted_sequence)):
    #                         predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")

    #                 actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

    #                 for a, p in zip(actual_sequence, predicted_sequence):
    #                     if "greater than mean" in a:
    #                         if "greater than mean" in p:
    #                             self.metrics["operation_exact_match"].update(1, 1)
    #                             col = a.split("value of")[1].split("is")[0]
    #                             if col in p:
    #                                 self.metrics["column_exact_match"].update(1, 1) 
    #                         else:
    #                             self.metrics["operation_exact_match"].update(0, 1)

    #                     if "greater than median" in a:
    #                         if "greater than median" in p:
    #                             self.metrics["operation_exact_match"].update(1, 1)
    #                             col = a.split("value of")[1].split("is")[0]
    #                             if col in p:
    #                                 self.metrics["column_exact_match"].update(1, 1) 
    #                         else:
    #                             self.metrics["operation_exact_match"].update(0, 1)

    #                     elif "weighted combination" in a:
    #                         if "weighted combination" in p:
    #                             self.metrics["operation_exact_match"].update(1, 1)
    #                             correct = 0
    #                             total = 0
    #                             columns = [col.strip() for col in a.split("combination of")[1].split("and")]
    #                             # if len(columns) == 2:
    #                             for col in columns:
    #                                 if col in p:
    #                                     correct += 1
    #                                 total += 1
    #                             self.metrics["column_exact_match"].update(correct, total)
    #                         else:
    #                             self.metrics["operation_exact_match"].update(0, 1)

    #                     elif "multiplying" in a:
    #                         if "multiplying" in p:
    #                             self.metrics["operation_exact_match"].update(1, 1)
    #                             columns = [col.strip() for col in a.split("multiplying")[1].split("and")]
    #                             correct = 0
    #                             total = 0
    #                             for col in columns:
    #                                 if col in p:
    #                                     correct += 1
    #                                 total += 1
    #                             self.metrics["column_exact_match"].update(correct, total)
    #                         else:
    #                             self.metrics["operation_exact_match"].update(0, 1)

    #                     else:
    #                         if "correlation" in p:
    #                             self.metrics["operation_exact_match"].update(1, 1)
    #                         else:
    #                             self.metrics["operation_exact_match"].update(0, 1)

    #     elif self.config.training.training_type == "table_question_answering":
    #         with tqdm(dataloader, unit = "batch", position = 0, leave = True, desc = "Metrics") as tepoch:
    #             for batch_idx, batch in enumerate(tepoch):
                    

    #                 if self.config.model.use_position_ids:
    #                     raise NotImplementedError
    #                     input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

    #                     # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
    #                     output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
    #                                                             max_length = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

    #                 else:
    #                     if self.config.model.type == "encoder-decoder":
    #                         input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
    #                         actual_output_ids = labels.clone()
    #                         output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                                 num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
    #                     elif self.config.model.type == "decoder-only":
    #                         input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
    #                         inputs_embeds = self.model.model.get_input_embeddings()(inference_input_ids.to(self.device))
    #                         # output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                         #                                     num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))
    #                         output_ids = self.model.model.generate(inputs_embeds = inputs_embeds, max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

    #                 predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #                 if self.config.model.type == "decoder-only":
    #                     for i in range(len(predicted_sequence)):
    #                         predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
                    
    #                 actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

    #                 for a, p in zip(actual_sequence, predicted_sequence):
    #                     pred = to_value_list([p])
    #                     gold = to_value_list([a])
    #                     correct = check_denotation(pred, gold)

    #                     if correct:
    #                         self.metrics["accuracy"].update(1, 1)
    #                     else:
    #                         self.metrics["accuracy"].update(0, 1)


    #     elif self.config.training.training_type == "table_decomposition":

    #         with tqdm(dataloader, unit = "batch", position = 0, leave = True, desc = "Metrics") as tepoch:
    #             for batch_idx, batch in enumerate(tepoch):
                    
    #                 if self.config.model.use_position_ids:
    #                     raise NotImplementedError
    #                     input_ids, attention_mask, token_type_ids, decoder_input_ids, position_ids, labels = batch

    #                     # model.module.generate(input_ids = input_ids, attention_mask = attention_mask, tokenizer = self.tokenizer)
    #                     output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), attention_mask = attention_mask.to(self.device), 
    #                                                             max_length = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, position_ids = position_ids.to(self.device))

    #                 else:
    #                     if self.config.model.type == "encoder-decoder":
    #                         input_ids, attention_mask, token_type_ids, decoder_input_ids, labels = batch
    #                         actual_output_ids = labels.clone()
    #                         output_ids = self.model.model.generate(input_ids = input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                                 num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(self.device))
    #                     elif self.config.model.type == "decoder-only":
    #                         input_ids, attention_mask, token_type_ids, inference_input_ids, inference_attention_mask, actual_output_ids, labels = batch
    #                         output_ids = self.model.model.generate(input_ids = inference_input_ids.to(self.device), max_new_tokens = self.config.tokenizer.output_max_length, 
    #                                                             num_beams = 3, early_stopping = True, attention_mask = inference_attention_mask.to(self.device))

    #                 predicted_sequence = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    #                 if self.config.model.type == "decoder-only":
    #                     for i in range(len(predicted_sequence)):
    #                         predicted_sequence[i] = predicted_sequence[i].replace(self.tokenizer.decode(inference_input_ids[i], skip_special_tokens = True), "")
                    
    #                 actual_sequence = self.tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

    #                 for a, p in zip(actual_sequence, predicted_sequence):

    #                     gold = a.split()
    #                     pred = p.split()

    #                     gold_freq = Counter(gold)
    #                     pred_freq = Counter(pred)

    #                     correct = sum(min(gold_freq[word], pred_freq[word]) for word in pred)
    #                     total = len(gold)

    #                     self.metrics["accuracy"].update(correct, total)