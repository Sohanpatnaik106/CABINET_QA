"""
    This file contains the definition of a PyTorch Dataset class for a "SciGen" dataset. The dataset contains 
    table and their corresponding descriptions from ML Scientific papers present in ArXiv
"""

# Import the required libraries
import copy
import torch
import random
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from typing import Any, List
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from transformers import AutoTokenizer


from .utils import is_maskable

"""
    A dataset class corresponding to the HuggingFace dataset "SciGen"
    SciGen is a dataset consisting of tables from ML scientific paper present in ArXiv and their corresponding descriptions
"""
class SciGenDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(SciGenDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
                                                       padding_side = self.config.tokenizer.padding_side)
        
        # Check for the presence of different special tokens in the tokenizer
        # If absent, initialise them as the eos token
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})

        # Process the huggingface dataset and obtain the relevant text input, text output and the list of tables
        self.text_input, self.table, self.text_output = self._process_dataset()

        # List of punctuation tokens required while masking tokens for the task of generative masked language modelling
        self.punctuation_tokens = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
                     "@", "#", "$", "%", "^", "&", "*", "+", "=", "_", "~", "`"]
        
        # Create a list of potential maskable words list used for generative MLM task
        self.maskable_words = []
        if self.config.training.training_type == "masked_language_modelling":
            self.tokenized_text = [self.tokenizer.tokenize(x)[:self.config.tokenizer.max_length] for x in self.text_input]
            with open(self.config.data.maskable_words_file, "rb") as f:
                self.maskable_words = pickle.load(f)
    
    """
        Function which takes a list of input tokens and returns the indices to be masked.
        This follows the standard whole words masking scheme introduced during the training of Bert based architectures
        Currently, this is implemented for tokenizers that follow the scheme mentioned below
            - If the tokenizer breaks a words into multiple tokens, it should add Ġ character at the beginning of the first token
    """
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
        # Iterate over the tokens to find the candidate indices to be masked. This loop is explicitly used to obtain the multiple indices of a word
        # required for whole word masking scheme
        for i, token in enumerate(input_tokens_copy):
            if token == self.tokenizer.special_tokens_map["bos_token"] or token == self.tokenizer.special_tokens_map["eos_token"] or token in self.punctuation_tokens:
                continue
            if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])       

        maskable_token_idx = []

        # Loop to combine the tokens into single word and check whether they can be masked. Moreover, current word masking also depends on the previous word. 
        # Logic for this mentioned in the report / paper
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

        # A list of 1s and 0s indicating which tokens should be masked
        mask_labels = [1 if i in maskable_token_idx else 0 for i in range(len(input_tokens))]
        return torch.tensor(mask_labels), occurrences[0] + 1

    """
        Function which takes text input and table as the input, along with the max length and text output
            - Return the tokenized form of text: Input IDs, Attention Mask and Token Type IDs whichever necessary
            - Logic begind the tokenization scheme
                -   For decoder-only models, the input to the model consists of both the input sequence as well as the output sequence
                    Therefore, when text output is provided to this function, input text as well as output text is fed to the tokenizer
                -   For encoder-decoder models, the tokenization scheme is standard
    """
    def _tokenize(self, text_input, table = None, max_length = 512, text_output = None):

        if text_output is not None:
            if self.config.tokenizer.special_table_tok:
                raise NotImplementedError
            else:
                if table is not None:
                    table = table + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output
                else:
                    text_input = text_input + f" {self.tokenizer.special_tokens_map['sep_token']} " + text_output

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
        
    """
        Function to process one sample of the SciGen dataset
            - Extract the table, paper title, table caption and table description
            - Processes the text and return the required text input, the table and text output
    """
    def _process_one_sample(self, data):

        paper = data["paper"]
        table_caption = data["table_caption"]
        table_column_names = eval(data["table_column_names"])
        
        table_content_values = eval(data["table_content_values"])

        # Part of code that handles different task requirements
        if self.config.training.training_type == "description_generation" or self.config.training.training_type == "masked_language_modelling":
            text_output = f"{data['text']}"
        elif self.config.training.training_type == "column_reasoning":
            text_output = f"{data['dependency']}"

        if self.config.training.use_title:
            text_input = f"{paper} {self.tokenizer.special_tokens_map['sep_token']} {table_caption}"
        else:
            text_input = f"{table_caption}"

        if self.config.training.training_type == "masked_language_modelling":
            text_input += f" {self.tokenizer.special_tokens_map['sep_token']} {text_output} "

        # Part of code to check whether the tokenizer has a special functionality to handle tables (specific to TAPEX based models)
        if self.config.tokenizer.special_table_tok:
            table = pd.DataFrame.from_dict({col: [table_content_values[j][i] for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        else:
            table = "[HEADER] " + " | ".join(table_column_names)
            for row in table_content_values:
                table += " [ROW] " + " | ".join(row) 

        return text_input, table, text_output

    """
        Function to iterate over the huggingface dataset and return the list of text input, table and text output as reqired for the model
    """
    def _process_dataset(self):

        processed_data = Parallel(n_jobs = -1)(
            delayed(self._process_one_sample)(data) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type]))
        )

        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        text_output = [x[2] for x in processed_data]

        return text_input, table, text_output

    """
        Function to obtain the length of the dataset
    """
    def __len__(self):
        return len(self.text_input)

    """
        Function that takes index as an input and requires the tokenized input, tokenized output 
            - It returns the input IDs, attention mask and token type IDs whichever necessary
            - Currently, the code is not implement to incorporate position IDs
    """
    def __getitem__(self, index) -> Any:

        # NOTE: The processing of input for encoder-decoder models is different from that of decoder-only models

        if self.config.model.type == "encoder-decoder":
            # Condition to check whether table needs to be used as an input to the model
            # NOTE: Ablation of the impact of table for several pre-training and downstream tasks
            if self.config.model.use_table:
                tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
            else:
                tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

            # Input processing for the tasks of description generation and column reasoning
            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning":
                tokenized_output = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)

                # The decoder input ids will be one token shifted right of the the labels
                # NOTE: So, check whether bos token is added to the start of the decoder output sequence, if not, add manually
                if tokenized_output["input_ids"][0][0] != self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"]):
                    decoder_input_ids = tokenized_output["input_ids"][0].clone()
                    decoder_input_ids[1:] = decoder_input_ids[:-1].clone()
                    decoder_input_ids[0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])
                else:
                    decoder_input_ids = tokenized_output["input_ids"][0]

                # Label is decoder input ids with one token shifted left i.e., bos token is not required
                labels = decoder_input_ids.clone()
                labels[:-1] = labels[1:].clone()
            
            # Input processing for the task of masked languge modelling
            # NOTE: We address the task of masked language modelling in a generative manner
            elif self.config.training.training_type == "masked_language_modelling":
                
                # Obtain the mask labels (0s and 1s)
                mask_labels, desc_idx = self._whole_word_mask(self.tokenized_text[index])
                # Get the list of indices that can be masked, however, add 2 to those indices for correct alignment of indices with the tokens
                mask_labels = torch.nonzero(mask_labels, as_tuple = True)[0] + 2

                # Select only those indices which fall inside the max length of the tokenizer
                mask_labels = mask_labels[mask_labels < self.config.tokenizer.input_max_length]
                if mask_labels.size()[0] >= self.config.data.masked_gen_length // 2:
                    mask_labels = mask_labels[:self.config.data.masked_gen_length // 2]

                # Now, create the output labels with the actual tokens in place of the mask tokens, however, only the tokens are arranged as the output sequence with sep tokens in between
                tokenized_output['input_ids'] = torch.ones(1, self.config.data.masked_gen_length, dtype = torch.long) * self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])
                tokenized_output["input_ids"][0][1:2*mask_labels.size()[0]:2] = tokenized_input["input_ids"][0][mask_labels]
                tokenized_output["input_ids"][0][2*mask_labels.size()[0] + 1:] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])
                tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                tokenized_input["input_ids"][0][mask_labels] = self.tokenizer.mask_token_id

                # Similar notion of labels being one token shifted left of the decoder input ids
                labels = tokenized_output["input_ids"][0].clone()
                labels[:-1] = labels[1:].clone()

                # Replace pad, mask, sep, and other tokens with -100 to ignore loss computation
                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
                labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])] = -100

        # Tokenizers of decoder only models do not add start token, add them explicitly
        elif self.config.model.type == "decoder-only":
            tokenized_output = {}
            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning":
                
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
                    

        if self.config.model.type == "encoder-decoder":
            if self.config.model.use_position_ids:
                position_ids = torch.tensor([i for i in range(tokenized_input["input_ids"].shape[1])], dtype = torch.long)
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), decoder_input_ids.squeeze(), position_ids, labels

            else:
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), decoder_input_ids.squeeze(), labels
            
        elif self.config.model.type == "decoder-only":

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning":
                output_text = self.text_output[index]
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