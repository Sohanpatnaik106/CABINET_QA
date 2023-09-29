import os
import re
import copy
import spacy
import torch
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from spacy import displacy
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet
from spacy.lang.en import English
from spacy.matcher import Matcher
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from .utils import is_maskable



from src.bart.tokenization_tapex import TapexTokenizer

class WikiTQReasoningDataset(Dataset):

    def __init__(self, dataset, config):
        super(WikiTQReasoningDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
                                                       padding_side = self.config.tokenizer.padding_side)

        # self.tokenizer = TapexTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
        #                                                padding_side = self.config.tokenizer.padding_side)
        

        # if self.config.model.soft_decomposition_model is not None:
        #     self.soft_decomposition_model = AutoModel.from_pretrained(self.config.model.soft_decomposition_model_path)
        #     self.soft_decomposition_tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.soft_decomposition_tokenizer_path)

            # self.soft_decomposition_model.to("cuda:0")
        
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": "<s>"})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})

        
        with open("datasets/wiki_tq_reason.pkl", "rb") as f:
            self.reasons = pickle.load(f)

        # self.soft_decomposed_table = None
        # if self.config.data.decompose_table:
        #     if os.path.exists(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl")):
        #         # self.soft_decomposed_table = pickle.load(os.path.join(self.config.data.soft_decomposition_data_path, "soft_decomposition.pkl"))
        #         with open(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl"), "rb") as f:
        #             self.soft_decomposed_table = pickle.load(f)


        self.text_input, self.table, self.text_output = self._process_dataset()
        
        # with open(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl"), "rb") as f:
        #     self.soft_decomposed_table = pickle.load(f)
        # self.table = self.soft_decomposed_table
        # if self.config.model.soft_decomposition_model is not None and self.soft_decomposed_table is None:
        #     del self.soft_decomposition_model
        #     if not os.path.exists(self.config.data.soft_decomposition_data_path):
        #         os.makedirs(self.config.data.soft_decomposition_data_path)
        #         # pickle.dump(self.table, os.path.join(self.config.data.soft_decomposition_data_path, "soft_decomposition.pkl"))
        #     with open(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl"), "wb") as f:
        #         pickle.dump(self.table, f)



        self.punctuation_tokens = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
                     "@", "#", "$", "%", "^", "&", "*", "+", "=", "_", "~", "`"]
        
        self.maskable_words = []
        if self.config.training.training_type == "masked_language_modelling":
            self.tokenized_text = [self.tokenizer.tokenize(x)[:self.config.tokenizer.max_length] for x in self.text_input]
            with open(self.config.data.maskable_words_file, "rb") as f:
                self.maskable_words = pickle.load(f)

    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        occurrences = [i for i, item in enumerate(input_tokens) if item == self.tokenizer.special_tokens_map["eos_token"]]   
        # TODO: Fix the IndexError when occurences is an empty list
        if len(occurrences) == 0:
            return torch.tensor([0 for i in range(len(input_tokens))]), -1
        
        input_tokens_copy = copy.deepcopy(input_tokens)[occurrences[0]+1:]

        cand_indexes = []
        for i, token in enumerate(input_tokens_copy):
            if token == self.tokenizer.special_tokens_map["bos_token"] or token == self.tokenizer.special_tokens_map["eos_token"] or token in self.punctuation_tokens:
                continue

            if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])       

        maskable_token_idx = []

        for i, word in enumerate(cand_indexes):
            word = "".join([input_tokens_copy[t] for t in cand_indexes[i]])
            word = word.replace("Ġ", "").replace("Â", "")

            prev_word = None
            if i != 0:
                prev_word = "".join([input_tokens_copy[t] for t in cand_indexes[i-1]])
                prev_word = prev_word.replace("Ġ", "").replace("Â", "")


            if is_maskable(word, self.maskable_words, prev_word = prev_word) and random.random() > 0.5:
                maskable_token_idx.extend(cand_indexes[i])

        maskable_token_idx = [token_idx + occurrences[0] for token_idx in maskable_token_idx]

        mask_labels = [1 if i in maskable_token_idx else 0 for i in range(len(input_tokens))]
        return torch.tensor(mask_labels), occurrences[0] + 1



    def _tokenize(self, text_input, table = None, max_length = 512, text_output = None):

        if text_output is not None:
            if self.config.tokenizer.special_table_tok:
                raise NotImplementedError
            else:
                if table is not None:
                    table = table + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output
                else:
                    text_input = text_input + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output
            # text_input = text_input + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output

        if self.config.tokenizer.special_table_tok:
            if table is not None:
                return self.tokenizer(table, text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)
            else: 
                return self.tokenizer(answer = text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)
        else:
            if table is not None:
                return self.tokenizer(text_input, table, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)
            else:
                return self.tokenizer(text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)


    def _process_one_sample(self, data, idx = None):

        # question = data["question"]
        # table_column_names = data["table"]["header"]
        # table_content_values = data["table"]["rows"]

        # answer = data["answers"]
        # answer_list = answers = [str(a).lower() for a in data["answers"]]
        # answer = f", ".join(answer).lower()


        question = self.dataset["question"][idx]
        table_dict = eval(self.dataset["table"][idx])
        table_column_names = table_dict["header"]
        table_content_values = table_dict["rows"]

        answer = eval(self.dataset["answers"][idx])
        answer_list = answers = [str(a).lower() for a in self.dataset["answers"]]
        answer = f", ".join(answer).lower()

        # output_text = self.reasons[idx]
        output_text = self.dataset["reason"][idx]

        input_text = f"question: {question} answer: {answer}. "


        if self.config.tokenizer.special_table_tok:
            
            # table_content_values = [self.expand_numbers(table_content_values[i]) for i in range(len(table_content_values))]

            # table_content_values = [[self.expand_numbers(table_content_values[i][j]) for j in range(len(table_content_values[i]))] for i in range(len(table_content_values))]

            # for i in range(table_content_values):
            #     for j in range(table_content_values[i]):
            #         table_content_values[i][j] = self.expand_numbers(table_content_values[i][j])

            table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})

            if self.config.data.decompose_table:
                relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                
                if self.config.training.training_type != "table_decomposition":
                    
                    if len(relevant_rows) > 0:
                        table = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table = table[relevant_columns]
                else:
                    if len(relevant_rows) > 0:
                        table_output = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table_output = table[relevant_columns]
            
        else:
            
            if self.config.data.decompose_table:
                table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
                relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                
                if self.config.training.training_type != "table_decomposition":
                    if len(relevant_rows) > 0:
                        table = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table = table[relevant_columns]

                    table_column_names = table.columns.tolist()
                    table_content_values = table.values.tolist()

                else:
                    if len(relevant_rows) > 0:
                        table_output = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table_output = table[relevant_columns]


            table = "[HEADER] " + " | ".join(table_column_names)
            for row_id, row in enumerate(table_content_values):
                table += f" [ROW] {row_id}: " + " | ".join(row) 

            if self.config.training.training_type == "table_decomposition":
                table_column_names_output = table_output.columns.tolist()
                table_content_values_output = table_output.values.tolist()

                table_output = "[HEADER] " + " | ".join(table_column_names_output)
                for row_id, row in enumerate(table_content_values_output):
                    table_output += f" [ROW] {row_id}: " + " | ".join(row)

        if self.config.training.training_type == "table_decomposition":
            return question, table, table_output
        else:
            return input_text, table, output_text

    

    def _process_dataset(self):

        # processed_data = Parallel(n_jobs = 1)(
        #     delayed(self._process_one_sample)(data, i) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])) if i < 1000
        # )

        processed_data = []
        for i in tqdm(range(len(self.dataset)), position = 0, leave = True, total = len(self.dataset)):
            processed_data.append(self._process_one_sample(data = None, idx=i))


        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        text_output = [x[2] for x in processed_data]

        return text_input, table, text_output

    def __len__(self):
        return len(self.text_input)

    # NOTE: Currently implemented for tapex tokenizer
    def _get_row_ids(self, tokenized_text):
        
        row_ids = []
        row_idx = 0
        # tokenized_text = self.tokenizer.tokenize(text)

        # if len(tokenized_text) != self.config.tokenizer.input_max_length:
        #     tokenized_text = tokenized_text[1:]

        for token in tokenized_text:
            if "row" in token:
                row_idx += 1
            
            if "</s>" in token:
                row_idx = 0

            row_ids.append(row_idx)

        return torch.tensor(row_ids).unsqueeze(0)


    def _get_col_ids(self, tokenized_text):
        
        col_idx = 0
        col_ids = []
        flag = False
        # tokenized_text = self.tokenizer.tokenize(text)

        # if len(tokenized_text) != self.config.tokenizer.input_max_length:
        #     tokenized_text = tokenized_text[1:]

        for token in tokenized_text:

            if "|" in token:
                col_ids.append(0)
                col_idx += 1
                continue
            
            if "row" in token or "col" in token or "</s>" in token:
                col_idx = 0
                col_ids.append(col_idx) 
                flag = False
                continue
            
            if ":" in token:
                col_ids.append(col_idx)
                flag = True
                col_idx = 1
                continue

            if flag:
                col_ids.append(col_idx)
                continue
            
            col_ids.append(col_idx)

        return torch.tensor(col_ids).unsqueeze(0)


    def __getitem__(self, index) -> Any:

        
        # NOTE: Currently the implementation of row embeddings, column embeddings and segment embeddings is available for encode-decoder models

        # NOTE: Permute the rows and columns randomly
        # self.table[index] = self.table[index].sample(frac = 1, axis = 1)

        if self.config.model.type == "encoder-decoder":
            if self.config.model.use_table:
                tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
            else:
                tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition" \
                    or self.config.training.training_type == "table_reasoning":
                tokenized_output = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)
                labels = tokenized_output["input_ids"][0].clone()
                
                if labels[0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"]):
                    labels[:-1] = labels[1:].clone()
                else:
                    tokenized_output["input_ids"][0][1:] = tokenized_output["input_ids"][0][:-1].clone()
                    tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100
                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])] = -100
                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])] = -100

            elif self.config.training.training_type == "masked_language_modelling":

                mask_labels, desc_idx = self._whole_word_mask(self.tokenized_text[index])
                mask_labels = torch.nonzero(mask_labels, as_tuple = True)[0] + 2

                # Select the elements from the original tensor based on the random indices
                mask_labels = mask_labels[mask_labels < self.config.tokenizer.input_max_length]
                if mask_labels.size()[0] >= self.config.data.masked_gen_length // 2:
                    mask_labels = mask_labels[:self.config.data.masked_gen_length // 2]

                tokenized_output['input_ids'] = torch.ones(1, self.config.data.masked_gen_length, dtype = torch.long) * self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])
                tokenized_output["input_ids"][0][1:2*mask_labels.size()[0]:2] = tokenized_input["input_ids"][0][mask_labels]
                tokenized_output["input_ids"][0][2*mask_labels.size()[0] + 1:] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])
                tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                tokenized_input["input_ids"][0][mask_labels] = self.tokenizer.mask_token_id

                labels = tokenized_output["input_ids"][0].clone()
                labels[:-1] = labels[1:].clone()
                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100

            if self.config.tokenizer.use_row_col_ids:
                tokenized_text = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"].squeeze(0))
                tokenized_input["row_ids"] = self._get_row_ids(tokenized_text = tokenized_text)
                tokenized_input["col_ids"] = self._get_col_ids(tokenized_text = tokenized_text)


        # Tokenizers of decoder only models do not add start token, add them explicitly
        elif self.config.model.type == "decoder-only":
            tokenized_output = {}
            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition" \
                    or self.config.training.training_type == "table_reasoning":
                
                if self.config.model.use_table:
                    tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length, text_output = self.text_output[index])
                    inference_tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
                else:
                    tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length, text_output = self.text_output[index])
                    inference_tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

                idx = (inference_tokenized_input["input_ids"][0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])).nonzero(as_tuple = True)[0]
                if len(idx) != 0:
                    idx = idx[0]
                    inference_tokenized_input["input_ids"][0] = inference_tokenized_input["input_ids"][0]
                    inference_tokenized_input["attention_mask"][0][:idx] = 0
                    inference_tokenized_input["attention_mask"][0][idx:] = 1

                padded_input = torch.ones(self.config.tokenizer.input_max_length, dtype = torch.long) * self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])
                padded_input[self.config.tokenizer.input_max_length - inference_tokenized_input["input_ids"][0].shape[0]:] = inference_tokenized_input["input_ids"][0]
                inference_tokenized_input["input_ids"][0] = padded_input

                labels = tokenized_input["input_ids"][0].clone()
                actual_output_ids = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)["input_ids"].squeeze()

                indices = (labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])).nonzero(as_tuple = True)[0]
                if len(indices) >= 2:
                    out_start, out_end = indices[0] + 1, indices[1]
                    labels[:out_start], labels[out_end:] = -100, -100
                elif len(indices) == 1:
                    out_start = indices[0] + 1
                    labels[:out_start] = -100
                else:
                    labels[:] = -100
                    labels[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])


                tokenized_input["input_ids"][0][1:] = tokenized_input["input_ids"][0].clone()[:-1]
                tokenized_input["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

            elif self.config.training.training_type == "masked_language_modelling":

                if self.config.model.use_table:
                    tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
                else:
                    tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

                tokenized_input["input_ids"][0][1:] = tokenized_input["input_ids"][0].clone()[:-1]
                tokenized_input["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                mask_labels, desc_idx = self._whole_word_mask(self.tokenized_text[index])
                mask_labels = torch.nonzero(mask_labels, as_tuple = True)[0] + 2

                # Select the elements from the original tensor based on the random indices
                mask_labels = mask_labels[mask_labels < self.config.tokenizer.input_max_length]
                if mask_labels.size()[0] >= self.config.data.masked_gen_length // 2:
                    mask_labels = mask_labels[:self.config.data.masked_gen_length // 2]

                eos_indices = (tokenized_input["input_ids"][0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])).nonzero(as_tuple = True)[0]
                if len(eos_indices) < 4:
                    # NOTE: No masking possible for this                    
                    labels = torch.ones(tokenized_input["input_ids"].shape[1], dtype = torch.long) * (-100)
                    labels[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])
                else:

                    out_start = eos_indices[3]
                    mask_labels = mask_labels[:(self.config.tokenizer.input_max_length - out_start) // 2]

                    labels = torch.ones(tokenized_input["input_ids"].shape[1], dtype = torch.long) * (-100)
                    labels[out_start:out_start + 2*mask_labels.size()[0]:2] = tokenized_input["input_ids"][0][mask_labels]
                    labels[:-1] = labels[1:].clone()

                    tokenized_input["input_ids"][0][mask_labels] = self.tokenizer.mask_token_id

                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100
                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])] = -100
                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])] = -100

                    # NOTE: Discuss whether this is correct
                    labels[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])

        # NOTE: Row and column ids is implemented only for encoder-decoder models
        if self.config.model.type == "encoder-decoder":
            if self.config.tokenizer.use_row_col_ids:
                position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), tokenized_input["row_ids"].squeeze(), tokenized_input["col_ids"].squeeze(), labels

            else:
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), labels
        
        elif self.config.model.type == "decoder-only":

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition" \
                    or self.config.training.training_type == "table_reasoning":
                if self.config.model.use_position_ids:
                    position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), position_ids, inference_tokenized_input["input_ids"].squeeze(), inference_tokenized_input["attention_mask"].squeeze(), actual_output_ids, labels

                else:
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), inference_tokenized_input["input_ids"].squeeze(), inference_tokenized_input["attention_mask"].squeeze(), actual_output_ids, labels
            
            else:
                if self.config.model.use_position_ids:
                    position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), position_ids, labels

                else:
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), labels

    def collate_fn(self, items):
        pass




class WikiTQReasoningWithoutAnswerDataset(Dataset):

    def __init__(self, dataset, config):
        super(WikiTQReasoningWithoutAnswerDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
                                                       padding_side = self.config.tokenizer.padding_side)

        # self.tokenizer = TapexTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
        #                                                padding_side = self.config.tokenizer.padding_side)
        

        # if self.config.model.soft_decomposition_model is not None:
        #     self.soft_decomposition_model = AutoModel.from_pretrained(self.config.model.soft_decomposition_model_path)
        #     self.soft_decomposition_tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.soft_decomposition_tokenizer_path)

            # self.soft_decomposition_model.to("cuda:0")
        
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": "<s>"})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})

        

        # self.soft_decomposed_table = None
        # if self.config.data.decompose_table:
        #     if os.path.exists(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl")):
        #         # self.soft_decomposed_table = pickle.load(os.path.join(self.config.data.soft_decomposition_data_path, "soft_decomposition.pkl"))
        #         with open(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl"), "rb") as f:
        #             self.soft_decomposed_table = pickle.load(f)


        self.text_input, self.table, self.text_output = self._process_dataset()
        
        # with open(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl"), "rb") as f:
        #     self.soft_decomposed_table = pickle.load(f)
        # self.table = self.soft_decomposed_table
        # if self.config.model.soft_decomposition_model is not None and self.soft_decomposed_table is None:
        #     del self.soft_decomposition_model
        #     if not os.path.exists(self.config.data.soft_decomposition_data_path):
        #         os.makedirs(self.config.data.soft_decomposition_data_path)
        #         # pickle.dump(self.table, os.path.join(self.config.data.soft_decomposition_data_path, "soft_decomposition.pkl"))
        #     with open(os.path.join(self.config.data.soft_decomposition_data_path, f"{self.data_type}_soft_decomposition.pkl"), "wb") as f:
        #         pickle.dump(self.table, f)



        self.punctuation_tokens = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
                     "@", "#", "$", "%", "^", "&", "*", "+", "=", "_", "~", "`"]
        
        self.maskable_words = []
        if self.config.training.training_type == "masked_language_modelling":
            self.tokenized_text = [self.tokenizer.tokenize(x)[:self.config.tokenizer.max_length] for x in self.text_input]
            with open(self.config.data.maskable_words_file, "rb") as f:
                self.maskable_words = pickle.load(f)

    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        occurrences = [i for i, item in enumerate(input_tokens) if item == self.tokenizer.special_tokens_map["eos_token"]]   
        # TODO: Fix the IndexError when occurences is an empty list
        if len(occurrences) == 0:
            return torch.tensor([0 for i in range(len(input_tokens))]), -1
        
        input_tokens_copy = copy.deepcopy(input_tokens)[occurrences[0]+1:]

        cand_indexes = []
        for i, token in enumerate(input_tokens_copy):
            if token == self.tokenizer.special_tokens_map["bos_token"] or token == self.tokenizer.special_tokens_map["eos_token"] or token in self.punctuation_tokens:
                continue

            if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])       

        maskable_token_idx = []

        for i, word in enumerate(cand_indexes):
            word = "".join([input_tokens_copy[t] for t in cand_indexes[i]])
            word = word.replace("Ġ", "").replace("Â", "")

            prev_word = None
            if i != 0:
                prev_word = "".join([input_tokens_copy[t] for t in cand_indexes[i-1]])
                prev_word = prev_word.replace("Ġ", "").replace("Â", "")


            if is_maskable(word, self.maskable_words, prev_word = prev_word) and random.random() > 0.5:
                maskable_token_idx.extend(cand_indexes[i])

        maskable_token_idx = [token_idx + occurrences[0] for token_idx in maskable_token_idx]

        mask_labels = [1 if i in maskable_token_idx else 0 for i in range(len(input_tokens))]
        return torch.tensor(mask_labels), occurrences[0] + 1



    def _tokenize(self, text_input, table = None, max_length = 512, text_output = None):

        if text_output is not None:
            if self.config.tokenizer.special_table_tok:
                raise NotImplementedError
            else:
                if table is not None:
                    table = table + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output
                else:
                    text_input = text_input + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output
            # text_input = text_input + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output

        if self.config.tokenizer.special_table_tok:
            if table is not None:
                return self.tokenizer(table, text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)
            else: 
                return self.tokenizer(answer = text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)
        else:
            if table is not None:
                return self.tokenizer(text_input, table, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)
            else:
                return self.tokenizer(text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                            padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                            max_length = max_length, return_tensors = self.config.tokenizer.return_tensors,
                            return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                            return_attention_mask = self.config.tokenizer.return_attention_mask)


    def _process_one_sample(self, idx = None):

        question = self.dataset["question"][idx]
        table_dict = eval(self.dataset["table"][idx])
        table_column_names = table_dict["header"]
        table_content_values = table_dict["rows"]

        answer = eval(self.dataset["answers"][idx])
        answer_list = answers = [str(a).lower() for a in self.dataset["answers"]]
        answer = f", ".join(answer).lower()


        output_text = self.dataset["reason"][idx]
        input_text = f"Question: {question} "


        if self.config.tokenizer.special_table_tok:
            
            # table_content_values = [self.expand_numbers(table_content_values[i]) for i in range(len(table_content_values))]

            # table_content_values = [[self.expand_numbers(table_content_values[i][j]) for j in range(len(table_content_values[i]))] for i in range(len(table_content_values))]

            # for i in range(table_content_values):
            #     for j in range(table_content_values[i]):
            #         table_content_values[i][j] = self.expand_numbers(table_content_values[i][j])

            table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})

            if self.config.data.decompose_table:
                relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                
                if self.config.training.training_type != "table_decomposition":
                    
                    if len(relevant_rows) > 0:
                        table = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table = table[relevant_columns]
                else:
                    if len(relevant_rows) > 0:
                        table_output = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table_output = table[relevant_columns]
            
        else:
            
            if self.config.data.decompose_table:
                table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
                relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                
                if self.config.training.training_type != "table_decomposition":
                    if len(relevant_rows) > 0:
                        table = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table = table[relevant_columns]

                    table_column_names = table.columns.tolist()
                    table_content_values = table.values.tolist()

                else:
                    if len(relevant_rows) > 0:
                        table_output = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table_output = table[relevant_columns]


            table = "[HEADER] " + " | ".join(table_column_names)
            for row_id, row in enumerate(table_content_values):
                table += f" [ROW] {row_id}: " + " | ".join(row) 

            if self.config.training.training_type == "table_decomposition":
                table_column_names_output = table_output.columns.tolist()
                table_content_values_output = table_output.values.tolist()

                table_output = "[HEADER] " + " | ".join(table_column_names_output)
                for row_id, row in enumerate(table_content_values_output):
                    table_output += f" [ROW] {row_id}: " + " | ".join(row)

        if self.config.training.training_type == "table_decomposition":
            return question, table, table_output
        else:
            return input_text, table, output_text

    

    def _process_dataset(self):

        # processed_data = Parallel(n_jobs = 1)(
        #     delayed(self._process_one_sample)(data, i) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])) if i < 1000
        # )

        processed_data = []
        for i in tqdm(range(len(self.dataset)), position = 0, leave = True, total = len(self.dataset)):
            processed_data.append(self._process_one_sample(i))


        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        text_output = [x[2] for x in processed_data]

        return text_input, table, text_output

    def __len__(self):
        return len(self.text_input)

    # NOTE: Currently implemented for tapex tokenizer
    def _get_row_ids(self, tokenized_text):
        
        row_ids = []
        row_idx = 0
        # tokenized_text = self.tokenizer.tokenize(text)

        # if len(tokenized_text) != self.config.tokenizer.input_max_length:
        #     tokenized_text = tokenized_text[1:]

        for token in tokenized_text:
            if "row" in token:
                row_idx += 1
            
            if "</s>" in token:
                row_idx = 0

            row_ids.append(row_idx)

        return torch.tensor(row_ids).unsqueeze(0)


    def _get_col_ids(self, tokenized_text):
        
        col_idx = 0
        col_ids = []
        flag = False
        # tokenized_text = self.tokenizer.tokenize(text)

        # if len(tokenized_text) != self.config.tokenizer.input_max_length:
        #     tokenized_text = tokenized_text[1:]

        for token in tokenized_text:

            if "|" in token:
                col_ids.append(0)
                col_idx += 1
                continue
            
            if "row" in token or "col" in token or "</s>" in token:
                col_idx = 0
                col_ids.append(col_idx) 
                flag = False
                continue
            
            if ":" in token:
                col_ids.append(col_idx)
                flag = True
                col_idx = 1
                continue

            if flag:
                col_ids.append(col_idx)
                continue
            
            col_ids.append(col_idx)

        return torch.tensor(col_ids).unsqueeze(0)


    def __getitem__(self, index) -> Any:

        
        # NOTE: Currently the implementation of row embeddings, column embeddings and segment embeddings is available for encode-decoder models

        # NOTE: Permute the rows and columns randomly
        # self.table[index] = self.table[index].sample(frac = 1, axis = 1)

        if self.config.model.type == "encoder-decoder":
            if self.config.model.use_table:
                tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
            else:
                tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition" \
                    or self.config.training.training_type == "table_reasoning":
                tokenized_output = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)
                labels = tokenized_output["input_ids"][0].clone()
                
                if labels[0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"]):
                    labels[:-1] = labels[1:].clone()
                else:
                    tokenized_output["input_ids"][0][1:] = tokenized_output["input_ids"][0][:-1].clone()
                    tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100
                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])] = -100
                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])] = -100

            elif self.config.training.training_type == "masked_language_modelling":

                mask_labels, desc_idx = self._whole_word_mask(self.tokenized_text[index])
                mask_labels = torch.nonzero(mask_labels, as_tuple = True)[0] + 2

                # Select the elements from the original tensor based on the random indices
                mask_labels = mask_labels[mask_labels < self.config.tokenizer.input_max_length]
                if mask_labels.size()[0] >= self.config.data.masked_gen_length // 2:
                    mask_labels = mask_labels[:self.config.data.masked_gen_length // 2]

                tokenized_output['input_ids'] = torch.ones(1, self.config.data.masked_gen_length, dtype = torch.long) * self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])
                tokenized_output["input_ids"][0][1:2*mask_labels.size()[0]:2] = tokenized_input["input_ids"][0][mask_labels]
                tokenized_output["input_ids"][0][2*mask_labels.size()[0] + 1:] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])
                tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                tokenized_input["input_ids"][0][mask_labels] = self.tokenizer.mask_token_id

                labels = tokenized_output["input_ids"][0].clone()
                labels[:-1] = labels[1:].clone()
                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100

            if self.config.tokenizer.use_row_col_ids:
                tokenized_text = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"].squeeze(0))
                tokenized_input["row_ids"] = self._get_row_ids(tokenized_text = tokenized_text)
                tokenized_input["col_ids"] = self._get_col_ids(tokenized_text = tokenized_text)


        # Tokenizers of decoder only models do not add start token, add them explicitly
        elif self.config.model.type == "decoder-only":
            tokenized_output = {}
            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition" \
                    or self.config.training.training_type == "table_reasoning":
                
                if self.config.model.use_table:
                    tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length, text_output = self.text_output[index])
                    inference_tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
                else:
                    tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length, text_output = self.text_output[index])
                    inference_tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

                idx = (inference_tokenized_input["input_ids"][0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])).nonzero(as_tuple = True)[0]
                if len(idx) != 0:
                    idx = idx[0]
                    inference_tokenized_input["input_ids"][0] = inference_tokenized_input["input_ids"][0]
                    inference_tokenized_input["attention_mask"][0][:idx] = 0
                    inference_tokenized_input["attention_mask"][0][idx:] = 1

                padded_input = torch.ones(self.config.tokenizer.input_max_length, dtype = torch.long) * self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])
                padded_input[self.config.tokenizer.input_max_length - inference_tokenized_input["input_ids"][0].shape[0]:] = inference_tokenized_input["input_ids"][0]
                inference_tokenized_input["input_ids"][0] = padded_input

                labels = tokenized_input["input_ids"][0].clone()
                actual_output_ids = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)["input_ids"].squeeze()

                indices = (labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])).nonzero(as_tuple = True)[0]
                if len(indices) >= 2:
                    out_start, out_end = indices[0] + 1, indices[1]
                    labels[:out_start], labels[out_end:] = -100, -100
                elif len(indices) == 1:
                    out_start = indices[0] + 1
                    labels[:out_start] = -100
                else:
                    labels[:] = -100
                    labels[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])


                tokenized_input["input_ids"][0][1:] = tokenized_input["input_ids"][0].clone()[:-1]
                tokenized_input["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

            elif self.config.training.training_type == "masked_language_modelling":

                if self.config.model.use_table:
                    tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
                else:
                    tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

                tokenized_input["input_ids"][0][1:] = tokenized_input["input_ids"][0].clone()[:-1]
                tokenized_input["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                mask_labels, desc_idx = self._whole_word_mask(self.tokenized_text[index])
                mask_labels = torch.nonzero(mask_labels, as_tuple = True)[0] + 2

                # Select the elements from the original tensor based on the random indices
                mask_labels = mask_labels[mask_labels < self.config.tokenizer.input_max_length]
                if mask_labels.size()[0] >= self.config.data.masked_gen_length // 2:
                    mask_labels = mask_labels[:self.config.data.masked_gen_length // 2]

                eos_indices = (tokenized_input["input_ids"][0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])).nonzero(as_tuple = True)[0]
                if len(eos_indices) < 4:
                    # NOTE: No masking possible for this                    
                    labels = torch.ones(tokenized_input["input_ids"].shape[1], dtype = torch.long) * (-100)
                    labels[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])
                else:

                    out_start = eos_indices[3]
                    mask_labels = mask_labels[:(self.config.tokenizer.input_max_length - out_start) // 2]

                    labels = torch.ones(tokenized_input["input_ids"].shape[1], dtype = torch.long) * (-100)
                    labels[out_start:out_start + 2*mask_labels.size()[0]:2] = tokenized_input["input_ids"][0][mask_labels]
                    labels[:-1] = labels[1:].clone()

                    tokenized_input["input_ids"][0][mask_labels] = self.tokenizer.mask_token_id

                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100
                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])] = -100
                    labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])] = -100

                    # NOTE: Discuss whether this is correct
                    labels[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["eos_token"])

        # NOTE: Row and column ids is implemented only for encoder-decoder models
        if self.config.model.type == "encoder-decoder":
            if self.config.tokenizer.use_row_col_ids:
                position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), tokenized_input["row_ids"].squeeze(), tokenized_input["col_ids"].squeeze(), labels

            else:
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), labels
        
        elif self.config.model.type == "decoder-only":

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition" \
                    or self.config.training.training_type == "table_reasoning":
                if self.config.model.use_position_ids:
                    position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), position_ids, inference_tokenized_input["input_ids"].squeeze(), inference_tokenized_input["attention_mask"].squeeze(), actual_output_ids, labels

                else:
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), inference_tokenized_input["input_ids"].squeeze(), inference_tokenized_input["attention_mask"].squeeze(), actual_output_ids, labels
            
            else:
                if self.config.model.use_position_ids:
                    position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), position_ids, labels

                else:
                    return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                            tokenized_input["token_type_ids"].squeeze(), labels

    def collate_fn(self, items):
        pass