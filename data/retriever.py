import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer

# from .dataloader import WikiTQDataset

import json
import argparse

# from ..utils import process_config
from easydict import EasyDict

from datasets import load_dataset


import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import cosine





import copy
import torch
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from transformers import AutoTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

# from .utils import is_maskable

import pandas as pd
import spacy
import re
from nltk.corpus import wordnet
import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.matcher import Matcher

from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


def is_number(token):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    return bool(re.match(pattern, token))

    
def preprocess_text(text):

    # Convert the text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Handle contractions
    words = [contraction_expansion(word) for word in words]

    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Remove empty and single-character words
    words = [word for word in words if len(word) > 1]

    return words


def is_maskable(token, maskable_words = None, prev_word = None):

    num = is_number(token)
    if prev_word is not None:
        prev_word = preprocess_text(prev_word)
    else:
        prev_word = []

    if len(prev_word) == 0:
        if num and random.random() >= 0.1:
            return True
    else:
        if num and random.random() >= 0.1 and prev_word[0] not in ["tabl", "figur"]:
            return True

    token = preprocess_text(token)
    if len(token) == 0:
        return False
    
    token = token[0]
    if token in maskable_words:
        return True

    return False

class WikiTQDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(WikiTQDataset, self).__init__()

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


        self.text_input, self.table, self.text_output = self._process_dataset()

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
        

    def _decompose_table(self, question, answers, table):
        columns = set()
        rows = set()

        # Step 1: Pick columns mentioned in the question
        for column in table.columns:
            if any(col_word.lower() in question.lower() for col_word in column.split()):
                columns.add(column)

        # Step 2: Pick rows with the answer
        for answer in answers:
            for index, row in table.iterrows():
                if any(ans_word.lower() in str(row).lower() for ans_word in answer.split()):
                    rows.add(index)

        # Step 3: Regular expressions
        regex_patterns = {
            'date': r'\d{4}-\d{2}-\d{2}',  # Example pattern for date format: YYYY-MM-DD
            'currency': r'\$\d+(?:,\d+)?(?:\.\d+)?',  # Example pattern for currency format: $X or $X.XX or $X,XXX.XX
        }
        for pattern_name, regex_pattern in regex_patterns.items():
            columns.update(table.columns[table[table.columns].astype(str).apply(lambda x: x.str.contains(regex_pattern)).any()])

        # Step 4: Similarity-based matching
        question_vector = table.columns.to_series().apply(lambda col: fuzz.token_set_ratio(question.lower(), col.lower())).values.reshape(1, -1)
        column_similarities = cosine_similarity(question_vector, table.columns.to_series().apply(lambda col: fuzz.token_set_ratio(question.lower(), col.lower())).values.reshape(1, -1))
        similar_columns = table.columns[column_similarities[0].argsort()[::-1][:3]]  # Select top 3 similar columns
        columns.update(similar_columns)

        # Step 5: Keyword extraction from the question
        # keywords = set()
        # nlp = spacy.load("en_core_web_lg")
        # for token in nlp(question):
        #     if token.pos_ in ['NOUN', 'PROPN']:  # Consider nouns and proper nouns as keywords
        #         keywords.add(token.lemma_.lower())
        # for keyword in keywords:
        #     # columns.update(table.columns[table.columns.str.lower().str.contains(keyword)])
        #     columns.update(table.columns[table.columns.str.lower().str.contains(keyword)].tolist())


        # Step 6: Statistical analysis
        statistics = {
            'max': table.max(),
            'min': table.min(),
            'mean': table.mean(numeric_only=True),
            'median': table.median(numeric_only=True)
        }
        for stat_name, stat_values in statistics.items():
            for column in stat_values.index:
                if stat_values[column] in answers:
                    columns.add(column)

        # Step 7: Synonym and antonym matching
        words = question.split()
        # synonym_antonym_threshold = 0.8  # Threshold for synonym/antonym matching
        # for word in words:
        #     synonyms = set()
        #     antonyms = set()
        #     for syn in wordnet.synsets(word):
        #         for lemma in syn.lemmas():
        #             synonyms.add(lemma.name())
        #             if lemma.antonyms():
        #                 antonyms.add(lemma.antonyms()[0].name())
        #     for column in table.columns:
        #         if not nlp(column).vector_norm:
        #             continue

        #         for syn in synonyms:
        #             if not nlp(syn).vector_norm:
        #                 continue

        #             if nlp(column).similarity(nlp(syn)) >= synonym_antonym_threshold:
        #                 columns.add(column)

        #         for ant in antonyms:
        #             if not nlp(ant).vector_norm:
        #                 continue

        #             if nlp(column).similarity(nlp(ant)) >= synonym_antonym_threshold:
        #                 columns.add(column)

        # Step 8: Pick rows where a group of continuous words from the question is present
        for index, row in table.iterrows():
            row_text = str(row).lower()
            is_present = any(" ".join(words[i:]) in row_text for i in range(len(words)))

            if is_present:
                rows.add(index)

        return list(rows), list(columns)

    def _process_one_sample(self, data):

        question = data["question"]
        table_column_names = data["table"]["header"]
        table_content_values = data["table"]["rows"]

        answer = data["answers"]
        answer_list = answers = [str(a).lower() for a in data["answers"]]
        answer = f" {self.tokenizer.special_tokens_map['sep_token']} ".join(answer)

        if self.config.tokenizer.special_table_tok:
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

                    table_column_names = relevant_columns
                    table_content_values = table.values.tolist()

                else:
                    if len(relevant_rows) > 0:
                        table_output = table.iloc[relevant_rows]
                    
                    elif len(relevant_columns) > 0:
                        table_output = table[relevant_columns]


            table = "[HEADER] " + " | ".join(table_column_names)
            for row in table_content_values:
                table += " [ROW] " + " | ".join(row) 

            if self.config.training.training_type == "table_decomposition":
                table_column_names_output = table_output.columns.tolist()
                table_content_values_output = table_output.values.tolist()

                table_output = "[HEADER] " + " | ".join(table_column_names_output)
                for row in table_content_values_output:
                    table_output += " [ROW] " + " | ".join(row)

        if self.config.training.training_type == "table_decomposition":
            return question, table, table_output
        else:
            return question, table, answer
    

    def _process_dataset(self):

        processed_data = Parallel(n_jobs = -1)(
            delayed(self._process_one_sample)(data) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type])) if i < 200
        )

        text_input = [x[0] for x in processed_data]
        table = [x[1] for x in processed_data]
        text_output = [x[2] for x in processed_data]

        return text_input, table, text_output

    def __len__(self):
        return len(self.text_input)

    def __getitem__(self, index) -> Any:


        if self.config.model.type == "encoder-decoder":
            if self.config.model.use_table:
                tokenized_input = self._tokenize(self.text_input[index], self.table[index], max_length = self.config.tokenizer.input_max_length)
            else:
                tokenized_input = self._tokenize(self.text_input[index], max_length = self.config.tokenizer.input_max_length)

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition":
                tokenized_output = self._tokenize(self.text_output[index], max_length = self.config.tokenizer.output_max_length)
                labels = tokenized_output["input_ids"][0].clone()
                
                if labels[0] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"]):
                    labels[:-1] = labels[1:].clone()
                else:
                    tokenized_output["input_ids"][0][1:] = tokenized_output["input_ids"][0][:-1].clone()
                    tokenized_output["input_ids"][0][0] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["bos_token"])

                # labels[labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] = -100
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

        # Tokenizers of decoder only models do not add start token, add them explicitly
        elif self.config.model.type == "decoder-only":
            tokenized_output = {}
            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition":
                
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
                        tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), position_ids, labels

            else:
                return tokenized_input["input_ids"].squeeze(), tokenized_input["attention_mask"].squeeze(), \
                        tokenized_input["token_type_ids"].squeeze(), tokenized_output["input_ids"].squeeze(), labels
            
        elif self.config.model.type == "decoder-only":

            if self.config.training.training_type == "description_generation" or self.config.training.training_type == "column_reasoning" \
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition":
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










def get_relevance_score(s1, s2, tokenizer = None, model = None, config = None):


    # Step 1: Parsing the Table
    # table_string = "[HEADER] Rank Cyclist Team Time UCI ProTour\nPoints [ROW] 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10\" | 40 [ROW] 2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30 [ROW] 3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25 [ROW] 4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20 [ROW] 5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15 [ROW] 6 | Denis Menchov (RUS) | Rabobank | s.t. | 11 [ROW] 7 | Samuel Sánchez (ESP) | Euskaltel-Euskadi | s.t. | 7 [ROW] 8 | Stéphane Goubert (FRA) | Ag2r-La Mondiale | + 2\" | 5 [ROW] 9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2\" | 3 [ROW] 10 | David Moncoutié (FRA) | Cofidis | + 2\" | 1"
    # table_rows = table_string.split("[ROW]")[1:]  # Skip the header row
    # table_columns = [column.strip() for column in table_rows[0].split("|")]
    
    table_string = s2
    table_rows = table_string.split("[ROW]")[1:]  # Skip the header row
    # table_columns = [column.strip() for column in table_rows[0].split("|")]
    table_columns = table_string.split("[ROW]")[0]

    print("Table columns: ", table_columns)
    
    # print(table_columns)
    # exit(0)

    # Step 2: Tokenization and Encoding
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')

    # question = "What is the team of Alejandro Valverde?"
    question = s1
    combined_text = question + ' ' + table_string

    combined_tokens = tokenizer.tokenize(combined_text, add_special_tokens = True)

    combined_input = tokenizer(combined_text, padding="max_length", truncation=True, return_tensors="pt")
    combined_outputs = model(**combined_input)

    # Step 3: Masking Question Tokens and Computing Relevance Scores
    question_tokens = tokenizer.tokenize(question)
    column_tokens = tokenizer.tokenize(table_columns)
    # print(question_tokens)

    # print(tokenizer.decode(combined_input["input_ids"][0]))
    # print(tokenizer.decode(combined_input["input_ids"][0]))
    # exit(0)

    relevance_scores = []
    table_embedding = combined_outputs.last_hidden_state[0, len(question_tokens) + 1:]


    for masked_token in question_tokens:
        # print(masked_token)
        masked_input_ids = combined_input["input_ids"].clone()
        masked_input_ids[0, combined_tokens.index(masked_token)] = tokenizer.mask_token_id
        # print(tokenizer.decode(masked_input_ids[0]))
        # exit(0)
        masked_input = {"input_ids": masked_input_ids, "attention_mask": combined_input["attention_mask"]}
        masked_outputs = model(**masked_input)

        # print(masked_outputs.last_hidden_state.shape)
        # print(tokenizer.decode(masked_input_ids[0, len(question_tokens) + 1:]))
        # exit(0)

        # Step 4: Cosine Similarity
        
        masked_embedding = masked_outputs.last_hidden_state[0, len(question_tokens) + 1:]
        # print(table_embedding.shape)
        # print(masked_embedding.shape)
        # exit(0)
        similarity = 1 - cosine_similarity(table_embedding.detach().numpy(), masked_embedding.detach().numpy())

        # print(type(similarity))
        # exit(0)

        relevance_scores.append(np.diag(similarity))

    relevance_scores = np.array(relevance_scores)

    relevance_scores = np.mean(relevance_scores, axis = 0)
    # print(relevance_scores.shape)

    # print(question_tokens)
    # print(column_tokens)
    # print(table_string)
    # print(len(question_tokens))

    # exit(0)

    # Step 5: Relevance Score Calculation
    cell_relevance_scores = []

    for row in table_rows:
        columns = row.split("|")
        # print(columns)
        # exit(0)
        row_scores = []
        for cell in columns:
            tokenized_cell = tokenizer.tokenize(cell.strip())
            if len(tokenized_cell) == 0:
                cell_scores = [0]
                
            else:
                # print(tokenized_cell)
                # print(combined_tokens.index(tokenized_cell[0]))
                # print(combined_tokens)
                # exit(0)
                cell_scores = [relevance_scores[combined_tokens.index(token) - (len(question_tokens) + 1)] \
                               if (combined_tokens.index(token) - (len(question_tokens) + 1)) < relevance_scores.shape[0] \
                                else 0 for token in tokenized_cell]
                
                # row_scores.append(1 - sum(cell_scores) / len(cell_scores))
            row_scores.append(np.mean(cell_scores))

        cell_relevance_scores.append(row_scores)

    cell_relevance_scores = np.array(cell_relevance_scores)

    return 1 / (1 + np.exp(-(0.5 - cell_relevance_scores / 0.07)))

    # Print the relevance scores for each cell
    # for i, row in enumerate(table_rows):
    #     columns = row.split("|")
    #     for j, cell in enumerate(columns):
    #         print(f"Cell ({i+1}, {j+1}): Relevance Score = {cell_relevance_scores[i]}")



    # Tokenize the input sequences
    s1_tokens = tokenizer.tokenize(s1)
    s2_tokens = tokenizer.tokenize(s2)

    # Add special tokens and convert to input IDs
    input_ids = tokenizer.encode_plus(s1, s2, add_special_tokens=True, return_tensors='pt')['input_ids']

    # Get the outputs from BERT for the combined sequences
    outputs = model(input_ids)
    s2_outputs = outputs.last_hidden_state[0, len(s1_tokens) + 2:len(input_ids[0]) - 1]

    # Compute the representations for each token of s2 without masking
    s2_representations = s2_outputs.detach().numpy()

    # Create a 2D matrix to store cosine similarities
    num_tokens_s1 = len(s1_tokens)
    num_tokens_s2 = len(s2_tokens)
    similarity_matrix = torch.zeros((num_tokens_s1, num_tokens_s2))

    # Compute cosine similarity for each token in s1 with each token in s2 (masked and non-masked)
    for i, token_s1 in enumerate(s1_tokens):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i + 1] = tokenizer.mask_token_id

        masked_outputs = model(masked_input_ids)
        masked_s2_outputs = masked_outputs.last_hidden_state[0, len(s1_tokens) + 2:len(input_ids[0]) - 1]

        masked_s2_representations = masked_s2_outputs.detach().numpy()

        # Compute cosine similarity between non-masked and masked representations
        similarities = cosine_similarity(s2_representations, masked_s2_representations)
        similarity_matrix[i] = torch.tensor(similarities[0])

    # Compute relevance scores for each token in s2
    relevance_scores = torch.mean(1 - similarity_matrix, dim=0)

    # Compute cell scores by aggregating relevance scores for each cell
    cell_scores = []
    current_cell_score = 0.0
    current_cell_token_count = 0
    
    print(s2_tokens)

    row_start = False
    for token_idx, token in enumerate(s2_tokens):

        current_cell_score += relevance_scores[token_idx]
        current_cell_token_count += 1

        if token == '|' or token == '\n':
            cell_scores.append(current_cell_score / current_cell_token_count)
            current_cell_score = 0.0
            current_cell_token_count = 0

    return cell_scores



    # Load the BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the input sequences
    question_tokens = tokenizer.tokenize(question)
    table_tokens = tokenizer.tokenize(table)

    print(question_tokens)
    print(len(question_tokens))
    print(table_tokens)
    print(len(table_tokens))


    # Add special tokens and convert to input IDs
    input_ids = tokenizer.encode_plus(question, table, add_special_tokens=True, return_tensors='pt')['input_ids']

    # Get the outputs from BERT for the combined sequences
    outputs = model(input_ids)
    table_outputs = outputs.last_hidden_state[0, len(question_tokens) + 2:len(input_ids[0]) - 1]

    # Compute the representations for each token of table without masking
    table_representations = table_outputs.detach().numpy()

    # Create a 2D matrix to store cosine similarities
    num_tokens_question = len(question_tokens)
    num_tokens_table = len(table_tokens)
    similarity_matrix = torch.zeros((num_tokens_question, num_tokens_table))

    # Compute cosine similarity for each token in question with each token in table (masked and non-masked)
    for i, token_question in enumerate(question_tokens):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i + 1] = tokenizer.mask_token_id

        masked_outputs = model(masked_input_ids)
        masked_table_outputs = masked_outputs.last_hidden_state[0, len(question_tokens) + 2:len(input_ids[0]) - 1]

        masked_table_representations = masked_table_outputs.detach().numpy()

        # print(table_representations.shape)
        # print(masked_table_representations.shape)
        # exit(0)
        # Compute cosine similarity between non-masked and masked representations
        similarities = cosine_similarity(table_representations, masked_table_representations)
        similarity_matrix[i] = torch.tensor(similarities[0])

    # Compute relevance scores for each token in s2
    relevance_scores = torch.mean(1 - similarity_matrix, dim=0)
    return relevance_scores

    # Compute average scores for each word in s2
    word_scores = []
    current_word_score = 0.0
    current_word_token_count = 0

    for token_idx, token in enumerate(question_tokens):
        current_word_score += relevance_scores[token_idx]
        current_word_token_count += 1

        if token.startswith("##"):
            continue

        word_scores.append(current_word_score / current_word_token_count)
        current_word_score = 0.0
        current_word_token_count = 0

    # Handle the last word if it ends with subword tokens
    if current_word_token_count > 0:
        word_scores.append(current_word_score / current_word_token_count)

    return word_scores


def process_config(config: dict, args = None):

    if args is not None:
        args_dict = vars(args)
        merged_dict = {**args_dict, **config}
    else:
        merged_dict = config

    merged_config = EasyDict(merged_dict)
    return merged_config




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/wiki_tq/t5.json", type = str, help = "Path to experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)

    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if config.data.config_name is not None:
        dataset = load_dataset(config.data.data_path, config.data.config_name)
    else:
        dataset = load_dataset(config.data.data_path)

    # train_dataset = WikiTQDataset(dataset, config, "train")
    # validation_dataset = WikiTQDataset(dataset, config, "validation")
    test_dataset = WikiTQDataset(dataset, config, "test")


    for i in range(len(test_dataset.text_input)):

        if i < 22:
            continue
        question = test_dataset.text_input[i]
        table = test_dataset.table[i]
        word_scores = get_relevance_score(question, table, tokenizer, model, config)

        print(question)
        print(table)

        print(word_scores.shape)
        print(len(word_scores))
        print(table.split())
        print(len(table.split()))
        break

