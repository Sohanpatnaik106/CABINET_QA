"""
    This file contains the definition of a PyTorch Dataset class for a "TabFact" dataset. The dataset contains 
    table and corresponding entailing or refuting statements
"""

# Import the required libraries
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from transformers import AutoTokenizer

"""
    A dataset class corresponding to the HuggingFace dataset "TabFact"
    TabFact is a dataset consisting of tables and certain statements corresponding to the tables. 
    The statements may be entailing or refuting w.r.t. the table
"""
class TabFactDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(TabFactDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type

        self.text_input, self.table, self.label = self._process_dataset()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only)

        # if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
        #     self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        # if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
        #     self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})


    def _tokenize(self, text_input, table = None):

        if table is not None:
            return self.tokenizer(table, text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                        padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                        max_length = self.config.tokenizer.max_length, return_tensors = self.config.tokenizer.return_tensors,
                        return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                        return_attention_mask = self.config.tokenizer.return_attention_mask)
        else: 
            return self.tokenizer(answer = text_input, add_special_tokens = self.config.tokenizer.add_special_tokens,
                        padding = self.config.tokenizer.padding, truncation = self.config.tokenizer.truncation, 
                        max_length = self.config.tokenizer.max_length, return_tensors = self.config.tokenizer.return_tensors,
                        return_token_type_ids = self.config.tokenizer.return_token_type_ids,
                        return_attention_mask = self.config.tokenizer.return_attention_mask)


    def _process_one_sample(self, data):
        text_input = data["statement"] + " <sep> " + data["table_caption"]

        # table_text is a string representation of the table
        table_text = data["table_text"]
        table_list = table_text.split("\n")

        table_column_names = table_list[0].split("#")
        table_content_values = [row.split("#") for row in table_list[1:]]
        table_content_values = [l for l in table_content_values if len(l) == len(table_column_names)]

        label = int(data["label"])

        # TODO: Implement for non special tokenizer for table

        if self.config.tokenizer.special_table_tok:
            table = pd.DataFrame.from_dict({col: [table_content_values[j][i] for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        else:
            table = "[HEADER] " + " ".join(table_column_names)
            for row in table_content_values:
                table += " [ROW] " + " | ".join(row) 
        
        # table = pd.DataFrame.from_dict({col: [table_content_values[j][i] for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        return text_input, table, label

    def _process_dataset(self):

        # processed_data = Parallel(n_jobs = self.config.system.num_workers)(
        #     delayed(self._process_one_sample)(data) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type]))
        # )

        processed_data = []
        for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])):
            processed_data.append(self._process_one_sample(data))

        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        label = [x[2] for x in processed_data]

        return text_input, table, label


    def __len__(self):
        return len(self.text_input)

    def __getitem__(self, index):

        if self.config.model.use_table:
            tokenized_input = self._tokenize(self.text_input[index], self.table[index])
        else:
            tokenized_input = self._tokenize(self.text_input[index])

        label = torch.tensor(self.label[index])

        return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                    tokenized_input["token_type_ids"].squeeze(), label

    def collate_fn(self, items):
        pass