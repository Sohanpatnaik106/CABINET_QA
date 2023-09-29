

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

# NOTE: This dataset class is not updated w.r.t. the latest experimental setting
class ToTToDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(ToTToDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only)
        
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": "<s>"})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})


        self.text_input, self.table, self.text_output = self._process_dataset()
        self.tokenized_text = [self.tokenizer.tokenize(x)[:self.config.tokenizer.max_length] for x in self.text_input]

        self.punctuation_tokens = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
                     "@", "#", "$", "%", "^", "&", "*", "+", "=", "_", "~", "`"]
        
        self.maskable_words = []
        if self.config.training.training_type == "masked_language_modelling":
            with open(self.config.data.maskable_words_file, "rb") as f:
                self.maskable_words = pickle.load(f)


    def _tokenize(self, text_input, table = None, max_length = 512):
        
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

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
            Get 0/1 labels for masked tokens with whole word mask proxy
        """

        occurrences = [i for i, item in enumerate(input_tokens) if item == "</s>"]   
        # TODO: Fix the IndexError when occurences is an empty list
        if len(occurrences) == 0:
            return torch.tensor([0 for i in range(len(input_tokens))]), -1
        
        input_tokens_copy = copy.deepcopy(input_tokens)[occurrences[0]+1:]

        cand_indexes = []
        for i, token in enumerate(input_tokens_copy):
            if token == "<s>" or token == "</s>" or token in self.punctuation_tokens:
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

            if is_maskable(word, self.maskable_words, prev_word = prev_word):
                maskable_token_idx.extend(cand_indexes[i])

        maskable_token_idx = [token_idx + occurrences[0] for token_idx in maskable_token_idx]

        mask_labels = [1 if i in maskable_token_idx else 0 for i in range(len(input_tokens))]
        return torch.tensor(mask_labels), occurrences[0] + 1

    def _process_one_sample(self, data):

        page_title = data["table_page_title"]
        section_title = data["table_section_title"]
        table = data["table"]
        target = data["target"]

        table_column_names = [[d["value"] for d in row if d["is_header"]] for row in table]
        table_content_values = [[d["value"] for d in row if not d["is_header"]] for row in table]

        if self.config.tokenizer.special_table_tok:
            # TODO: Think of implementing this as the table is hierarchical, so it's difficult to represent it as a dataframe
            table = pd.DataFrame.from_dict({col: [table_content_values[j][i] for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        else:
            table = ""
            for row in table_column_names:
                if len(row) == 0:
                    continue
                table += "[HEADER] " + " ".join(row) + " "
            
            for row in table_content_values:
                if len(row) == 0:
                    continue
                table += "[ROW] " + " | ".join(row) + " "

        text_input = page_title + " </s> " + section_title

        return text_input, table, target

    def _process_dataset(self):

        # processed_data = Parallel(n_jobs = -1)(
        #     delayed(self._process_one_sample)(data) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])) if i < 200
        # )
        
        processed_data = []
        for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])):
            if i < 200:
                processed_data.append(self._process_one_sample(data))
            else:
                break

        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        label = [x[2] for x in processed_data]

        return text_input, table, label


    def __len__(self):
        return len(self.text_input)

    def __getitem__(self, index) -> Any:

        if self.config.model.use_table:
            tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
        else:
            tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.output_max_length)

        tokenized_output = self._tokenize(self.text_output[index])

        if self.config.training.training_type == "masked_language_modelling":

            mask_labels, desc_idx = self._whole_word_mask(self.tokenized_text[index])

            mask_labels = torch.nonzero(mask_labels, as_tuple = True)[0]
            # Determine the number of values to sample (15% of total values)
            num_values = mask_labels.numel()
            num_samples = int(0.7 * num_values)

            # Generate random indices to sample from the tensor
            indices = torch.randperm(num_values)[:num_samples]

            # Select the elements from the original tensor based on the random indices
            mask_labels = mask_labels.view(-1)[indices].sort().values + 1
            mask_labels = mask_labels[mask_labels < 512]

            if desc_idx != -1:
                output_mask_labels = mask_labels - desc_idx
            else:
                output_mask_labels = mask_labels

            tokenized_input["input_ids"][0][mask_labels] = self.tokenizer.mask_token_id
            tokenized_output["input_ids"][0][torch.logical_not(torch.isin(torch.arange(tokenized_output["input_ids"][0].size(0)), torch.tensor(output_mask_labels)))] = -100

        return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                    tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze()

    def collate_fn(self, items):
        pass



class ToTToCellHighlightingDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(ToTToCellHighlightingDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only)

        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": "<s>"})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})
        
        self.text_input, self.table, self.text_output = self._process_dataset()
        self.tokenized_text = [self.tokenizer.tokenize(x)[:self.config.tokenizer.max_length] for x in self.text_input]

        self.punctuation_tokens = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
                     "@", "#", "$", "%", "^", "&", "*", "+", "=", "_", "~", "`"]
        
        self.maskable_words = []
        if self.config.training.training_type == "masked_language_modelling":
            with open(self.config.data.maskable_words_file, "rb") as f:
                self.maskable_words = pickle.load(f)


    def _tokenize(self, text_input, table = None, max_length = 512):
        
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


    def _add_adjusted_col_offsets(self, table):
        """Add adjusted column offsets to take into account multi-column cells."""
        adjusted_table = []
        for row in table:
            real_col_index = 0
            adjusted_row = []
            for cell in row:
                adjusted_cell = copy.deepcopy(cell)
                adjusted_cell["adjusted_col_start"] = real_col_index
                adjusted_cell["adjusted_col_end"] = (
                    adjusted_cell["adjusted_col_start"] + adjusted_cell["column_span"]
                )
                real_col_index += adjusted_cell["column_span"]
                adjusted_row.append(adjusted_cell)
            adjusted_table.append(adjusted_row)
        return adjusted_table


    def _get_heuristic_col_headers(self, adjusted_table, row_index, col_index):
        """Heuristic to find column headers."""
        adjusted_cell = adjusted_table[row_index][col_index]
        adjusted_col_start = adjusted_cell["adjusted_col_start"]
        adjusted_col_end = adjusted_cell["adjusted_col_end"]
        col_headers = []
        for r in range(0, row_index):
            row = adjusted_table[r]
            for cell in row:
                if (
                    cell["adjusted_col_start"] < adjusted_col_end
                    and cell["adjusted_col_end"] > adjusted_col_start
                ):
                    if cell["is_header"]:
                        col_headers.append(cell)

        return col_headers


    def get_totto_full_table(self, table, cell_indices, table_page_title = None, table_section_title = None):

        """Verbalize full table and return a string."""
        table_str = "Start of a new table with repetition of column names in between for your reference\n"
        if table_page_title:
            table_str += "<page_title> " + table_page_title + " </page_title> "
        if table_section_title:
            table_str += "<section_title> " + table_section_title + " </section_title> "

        adjusted_table = self._add_adjusted_col_offsets(table)

        col_headers = []
        for r_index, row in enumerate(table):
            row_str = "<row> "
            for c_index, col in enumerate(row):
                col_header = self._get_heuristic_col_headers(adjusted_table, r_index, c_index)
                
                if r_index == 1:
                    for ch in col_header:
                        if ch["value"] not in col_headers:
                            col_headers.append(ch["value"])


        highlighted_cells = []
        table_dict = {"header": col_headers, "rows": []}
        for r_index, row in enumerate(table):
            
            if r_index == 0:
                continue

            row_list = []
            for c_index, col in enumerate(row):
                
                # Select the highlighted cell
                if [r_index, c_index] in cell_indices:
                    highlighted_cells.append(col["value"])

                # The value of the cell.
                row_list.append(col["value"])


            table_dict["rows"].append(row_list)

        return table_dict, highlighted_cells

    def _process_one_sample(self, data):

        page_title = data["table_page_title"]
        section_title = data["table_section_title"]
        table = data["table"]
        target = data["target"]

        cell_indices = data["highlighted_cells"]

        # table_column_names = [[d["value"] for d in row if d["is_header"]] for row in table]
        # table_content_values = [[d["value"] for d in row if not d["is_header"]] for row in table]

        table_dict, highlighted_cells = self.get_totto_full_table(table=table, cell_indices=cell_indices)
        table_column_names = table_dict["header"]
        table_content_values = table_dict["rows"]

        if self.config.tokenizer.special_table_tok:
            # TODO: Think of implementing this as the table is hierarchical, so it's difficult to represent it as a dataframe
            table = pd.DataFrame.from_dict({col: [table_content_values[j][i] for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        else:
            table = "[HEADER] " + " | ".join(table_column_names)
            for row_id, row in enumerate(table_content_values):
                table += f" [ROW] {row_id}: " + " | ".join(row) 

        text_input = target
        text_output = ", ".join(highlighted_cells).strip()

        return text_input, table, text_output

    def _process_dataset(self):

        processed_data = Parallel(n_jobs = -1)(
            delayed(self._process_one_sample)(data) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type]))
        )

        # processed_data = []
        # for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])):
        #     if i < 200:
        #         processed_data.append(self._process_one_sample(data))
        #     else:
        #         break

        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        label = [x[2] for x in processed_data]

        return text_input, table, label


    def __len__(self):
        return len(self.text_input)

    def __getitem__(self, index) -> Any:

        if self.config.model.use_table:
            tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
        else:
            tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.output_max_length)

        tokenized_output = self._tokenize(self.text_output[index])
        labels = tokenized_output["input_ids"][0].clone()
                
        if labels[0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"]):
            labels[:-1] = labels[1:].clone()
        else:
            tokenized_output["input_ids"][0][1:] = tokenized_output["input_ids"][0][:-1].clone()
            tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

        labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100

        return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                    tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), labels

    def collate_fn(self, items):
        pass