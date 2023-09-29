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

import pickle

from src.bart.tokenization_tapex import TapexTokenizer



class WikiTQWithReasonAsOutputDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(WikiTQWithReasonAsOutputDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
                                                       padding_side = self.config.tokenizer.padding_side)
        
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})

        if self.data_type == "train":
            with open("datasets/wiki_tq_no_answer_in_reason_flant5.pkl", "rb") as f:
                self.reasons = pickle.load(f)

        elif self.data_type == "test":
            with open("datasets/test_wiki_tq_no_answer_in_reason_flant5.pkl", "rb") as f:
                self.reasons = pickle.load(f)

        else:
            self.reasons = [""] * len(self.dataset[self.data_type])

        self.text_input, self.table, self.text_output = self._process_dataset()


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

        question = data["question"]
        table_column_names = data["table"]["header"]
        table_content_values = data["table"]["rows"]

        answer = data["answers"]
        answer_list = answers = [str(a).lower() for a in data["answers"]]
        answer = f", ".join(answer).lower()

        reason = self.reasons[idx]

        text_input = f"question: {question}"
        text_output = f"reason: {reason} answer: {answer}"

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
            return text_input, table, text_output
    

    def _process_dataset(self):

        # processed_data = Parallel(n_jobs = 1)(
        #     delayed(self._process_one_sample)(data, i) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])) if i < 1000
        # )

        processed_data = []
        for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])):
            processed_data.append(self._process_one_sample(data, i))


        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        text_output = [x[2] for x in processed_data]

        return text_input, table, text_output

    def __len__(self):
        return len(self.text_input)


    def __getitem__(self, index) -> Any:

        # NOTE: Current implementation only for encoder-decoder type models

        tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
        
        tokenized_output = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)
        labels = tokenized_output["input_ids"][0].clone()
        
        if labels[0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"]):
            labels[:-1] = labels[1:].clone()
        else:
            tokenized_output["input_ids"][0][1:] = tokenized_output["input_ids"][0][:-1].clone()
            tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

        labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100

        return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), tokenized_input["token_type_ids"].squeeze(), \
                tokenized_output["input_ids"].squeeze(), labels