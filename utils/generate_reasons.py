
import os
os.chdir("../")

from src import T5ModelForTableReasoning
# from data import SequentialQADataset
from utils import process_config
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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




from src.bart.tokenization_tapex import TapexTokenizer


class SequentialQADataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(SequentialQADataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
                                                       padding_side = self.config.tokenizer.padding_side)

        # self.tokenizer = TapexTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
        #                                                padding_side = self.config.tokenizer.padding_side)
        

        # if self.config.model.soft_decomposition_model is not None:
        #     self.soft_decomposition_model = AutoModel.from_pretrained(self.config.model.soft_decomposition_model_path)
        #     self.soft_decomposition_tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.soft_decomposition_tokenizer_path)

            # self.soft_decomposition_model.to("cuda:0")
        
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.special_tokens_map["eos_token"]})

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
        return
        # """
        # Get 0/1 labels for masked tokens with whole word mask proxy
        # """

        # occurrences = [i for i, item in enumerate(input_tokens) if item == self.tokenizer.special_tokens_map["eos_token"]]   
        # # TODO: Fix the IndexError when occurences is an empty list
        # if len(occurrences) == 0:
        #     return torch.tensor([0 for i in range(len(input_tokens))]), -1
        
        # input_tokens_copy = copy.deepcopy(input_tokens)[occurrences[0]+1:]

        # cand_indexes = []
        # for i, token in enumerate(input_tokens_copy):
        #     if token == self.tokenizer.special_tokens_map["bos_token"] or token == self.tokenizer.special_tokens_map["eos_token"] or token in self.punctuation_tokens:
        #         continue

        #     if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
        #         cand_indexes[-1].append(i)
        #     else:
        #         cand_indexes.append([i])       

        # maskable_token_idx = []

        # for i, word in enumerate(cand_indexes):
        #     word = "".join([input_tokens_copy[t] for t in cand_indexes[i]])
        #     word = word.replace("Ġ", "").replace("Â", "")

        #     prev_word = None
        #     if i != 0:
        #         prev_word = "".join([input_tokens_copy[t] for t in cand_indexes[i-1]])
        #         prev_word = prev_word.replace("Ġ", "").replace("Â", "")


        #     if is_maskable(word, self.maskable_words, prev_word = prev_word) and random.random() > 0.5:
        #         maskable_token_idx.extend(cand_indexes[i])

        # maskable_token_idx = [token_idx + occurrences[0] for token_idx in maskable_token_idx]

        # mask_labels = [1 if i in maskable_token_idx else 0 for i in range(len(input_tokens))]
        # return torch.tensor(mask_labels), occurrences[0] + 1



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


    def _soft_decompose_table(self, s1, s2):
        
        # Step 1: Parsing the Table        
        table_string = s2
        table_rows = table_string.split("[ROW]")[1:]  # Skip the header row
        table_columns = table_string.split("[ROW]")[0]

        table_columns_list = [str(col).lower() for col in table_columns.replace("[HEADER]", "").strip().split("|")]
        table_columns_list = [name.strip() for name in table_columns_list]
    
        question = s1
        combined_text = question + ' ' + table_string

        combined_tokens = self.soft_decomposition_tokenizer.tokenize(combined_text, add_special_tokens = True, padding = "max_length", truncation = True)

        combined_input = self.soft_decomposition_tokenizer(combined_text, padding="max_length", truncation=True, return_tensors="pt")
        # combined_input = {"input_ids": combined_input["input_ids"].to("cuda:0"), "attention_mask": combined_input["attention_mask"].to("cuda:0")}
        # combined_input["input_ids"] = combined_input["input_ids"].to("cuda:0")
        # combined_input["attention_mask"] = combined_input["attention_mask"].to("cuda:0")

        with torch.no_grad():
            combined_outputs = self.soft_decomposition_model(**combined_input)

        # Step 3: Masking Question Tokens and Computing Relevance Scores
        question_tokens = self.soft_decomposition_tokenizer.tokenize(question)
        # column_tokens = self.soft_decomposition_tokenizer.tokenize(table_columns)

        relevance_scores = []
        table_embedding = combined_outputs.last_hidden_state[0, len(question_tokens) + 1:].detach().cpu().numpy()


        for masked_token in question_tokens:
            masked_input_ids = combined_input["input_ids"].clone()
            masked_input_ids[0, combined_tokens.index(masked_token)] = self.soft_decomposition_tokenizer.mask_token_id
            masked_input = {"input_ids": masked_input_ids, "attention_mask": combined_input["attention_mask"]}
            with torch.no_grad():
                masked_outputs = self.soft_decomposition_model(**masked_input)

            # Step 4: Cosine Similarity
            masked_embedding = masked_outputs.last_hidden_state[0, len(question_tokens) + 1:].detach().numpy()
            similarity = 1 - cosine_similarity(table_embedding, masked_embedding)
            relevance_scores.append(np.diag(similarity))

            del masked_outputs
            del masked_input_ids
            del masked_input

        relevance_scores = np.mean(np.array(relevance_scores), axis = 0)

        # Step 5: Relevance Score Calculation
        # cell_relevance_scores = []
        cell_relevance_scores = np.zeros((len(table_rows), len(table_columns_list)))

        for i, row in enumerate(table_rows):
            columns = row.split("|")
            # row_scores = []
            for j, cell in enumerate(columns):
                tokenized_cell = self.soft_decomposition_tokenizer.tokenize(cell.strip())
                if len(tokenized_cell) == 0:
                    cell_scores = [0]
                    
                else:
                    cell_scores = [relevance_scores[combined_tokens.index(token) - (len(question_tokens) + 1)] \
                                if token in combined_tokens and (combined_tokens.index(token) - (len(question_tokens) + 1)) < relevance_scores.shape[0] \
                                    else 0 for token in tokenized_cell]
                    
                # row_scores.append(np.mean(cell_scores))
                cell_relevance_scores[i, j] = np.mean(cell_scores)

            # cell_relevance_scores.append(row_scores)

        cell_relevance_scores = 1 / (1 + np.exp(-(0.5 - np.array(cell_relevance_scores) / 0.07)))

        # Pick the top k relevant columns and rows using the cell relevance scores
        flattened_relevance_scores = cell_relevance_scores.flatten()
        topk = min(self.config.data.topk_cells, flattened_relevance_scores.shape[0])

        # if self.config.data.topk_cells > flattened_relevance_scores.shape[0]:
        #     topk = flattened_relevance_scores.shape[0]
        # else:
        #     topk = self.config.data.topk_cells
        top_k_indices = np.argsort(flattened_relevance_scores)[-topk:]

        # Get the row and column indices of the top k cells
        row_indices, col_indices = np.unravel_index(top_k_indices, cell_relevance_scores.shape)

        # Get the corresponding column names and row indices
        top_k_column_names = [table_columns_list[col_idx] for col_idx in col_indices]
        top_k_row_indices = row_indices.tolist()

        return list(set(top_k_row_indices)), list(set(top_k_column_names))


    def expand_numbers(self, strings):
        # Define lookup tables for various place values
        units = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        scales = ['', 'thousand', 'lakh', 'crore']

        expanded_strings = []
        for string in strings:
            match = re.search(r'\d{1,10}(?:,\d{3})*|\d+', string)  # Regular expression to match numbers
            if match:
                number = match.group()
                number = number.replace(',', '')  # Remove commas if present
                number = int(number)
                if number == 0:
                    expanded_string = 'zero'
                else:
                    num_parts = []
                    scale_count = 0
                    while number > 0:
                        num_part = number % 1000  # Consider the last 3 digits
                        if num_part > 0:
                            num_part_words = []
                            hundreds_digit = num_part // 100
                            if hundreds_digit > 0:
                                num_part_words.append(units[hundreds_digit])
                                num_part_words.append('hundred')
                            tens_digit = (num_part // 10) % 10
                            units_digit = num_part % 10
                            if tens_digit == 1:
                                num_part_words.append(teens[units_digit])
                            else:
                                if tens_digit > 1:
                                    num_part_words.append(tens[tens_digit])
                                if units_digit > 0:
                                    num_part_words.append(units[units_digit])
                            num_parts.append(' '.join(num_part_words) + ' ' + scales[scale_count])
                        number //= 1000
                        scale_count += 1
                    num_parts.reverse()
                    expanded_string = ' '.join(num_parts)
                expanded_string = string.replace(match.group(), expanded_string)  # Replace number in string
                expanded_strings.append(expanded_string)
            else:
                expanded_strings.append(string)
        return expanded_strings


    def _process_one_sample(self, data, idx = None):

        # question = data["question"]
        # question = " ".join(data["question_and_history"])
        question = " ".join(data["question"])

        table_column_names = data["table_header"]
        table_content_values = data["table_data"]

        answer = data["answer_text"]
        answer_list = answers = [str(a).lower() for a in data["answer_text"]]
        # answer = f" {self.tokenizer.special_tokens_map['sep_token']} ".join(answer)
        answer = ", ".join(answer).lower()


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
            return question, table, answer

    

        # table = None

        # table = self.soft_decomposed_table[idx]

        # if self.config.tokenizer.special_table_tok:
            
        #     if self.soft_decomposed_table is not None:
        #         if self.config.training.training_type != "table_decomposition":
        #             table = self.soft_decomposed_table[idx]
        #         else:
        #             table_output = self.soft_decomposed_table[idx]
            
        #     else:
        #         table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})

        #         if self.config.data.decompose_table:
        #             if self.config.model.soft_decomposition_model is not None:
        #                 table_string = "[HEADER] " + " | ".join(table_column_names)
        #                 for row in table_content_values:
        #                     table_string += " [ROW] " + " | ".join(row)
        #                 relevant_rows, relevant_columns = self._soft_decompose_table(question, table_string)

        #                 # print(relevant_rows)
        #                 # print(relevant_columns)
                        
        #             else:
        #                 relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                    
        #             if self.config.training.training_type != "table_decomposition":
                        
        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table = table[relevant_columns].iloc[relevant_rows]
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table = table[relevant_columns]
        #             else:
        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table_output = table_output[relevant_columns].iloc[relevant_rows]
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table_output = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table_output = table[relevant_columns]
            
        # else:
            
        #     if self.soft_decomposed_table is not None:
        #         if self.config.training.training_type != "table_decomposition":
        #             table = self.soft_decomposed_table[idx]
        #         else:
        #             table_output = self.soft_decomposed_table[idx]

        #     else:
        #         if self.config.data.decompose_table:
        #             table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        #             if self.config.model.soft_decomposition_model is not None:
        #                 table_string = "[HEADER] " + " | ".join(table_column_names)
        #                 for row in table_content_values:
        #                     table_string += " [ROW] " + " | ".join(row)
        #                 relevant_rows, relevant_columns = self._soft_decompose_table(question, table_string)
                        
        #             else:
        #                 relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                    
        #             if self.config.training.training_type != "table_decomposition":

        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table = table[relevant_columns].iloc[relevant_rows]
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table = table[relevant_columns]

        #                 table_column_names = table.column.tolist()
        #                 table_content_values = table.values.tolist()

        #             else:
        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table_output = table[relevant_columns].iloc[relevant_rows]
                        
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table_output = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table_output = table[relevant_columns]


        #     table = "[HEADER] " + " | ".join(table_column_names)
        #     for row in table_content_values:
        #         table += " [ROW] " + " | ".join(row) 

        #     if self.config.training.training_type == "table_decomposition":
        #         table_column_names_output = table_output.columns.tolist()
        #         table_content_values_output = table_output.values.tolist()

        #         table_output = "[HEADER] " + " | ".join(table_column_names_output)
        #         for row in table_content_values_output:
        #             table_output += " [ROW] " + " | ".join(row)

        # # if self.config.model.soft_decomposition_model is not None:
        # #     if self.config.tokenizer.special_table_tok:
        # #         raise NotImplementedError
            
        # #     cell_relevance_score = self._get_soft_decomposition_relevance_score(question, table)
        # #     return question, table, answer, cell_relevance_score

        # # else:

        # return question, table, answer

        # if self.config.training.training_type == "table_decomposition":
        #     return question, table, table_output
        # else:
        #     return question, table, answer
    

    def _process_dataset(self):

        # processed_data = Parallel(n_jobs = -1)(
        #     delayed(self._process_one_sample)(data, i) for i, data in tqdm(enumerate(self.dataset[self.data_type]), position = 0, leave = True, total = len(self.dataset[self.data_type]))
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
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition":
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


class WikiSQLDataset(Dataset):

    def __init__(self, dataset, config, data_type):
        super(WikiSQLDataset, self).__init__()

        self.dataset = dataset
        self.config = config
        self.data_type = data_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
                                                       padding_side = self.config.tokenizer.padding_side)

        # self.tokenizer = TapexTokenizer.from_pretrained(self.config.tokenizer.tokenizer_path, local_files_only = self.config.tokenizer.local_files_only,
        #                                                padding_side = self.config.tokenizer.padding_side)
        

        # if self.config.model.soft_decomposition_model is not None:
        #     self.soft_decomposition_model = AutoModel.from_pretrained(self.config.model.soft_decomposition_model_path)
        #     self.soft_decomposition_tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer.soft_decomposition_tokenizer_path)

            # self.soft_decomposition_model.to("cuda:0")
        
        if "bos_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "pad_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "sep_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"sep_token": self.tokenizer.special_tokens_map["eos_token"]})

        if "mask_token" not in list(self.tokenizer.special_tokens_map.keys()):
            self.tokenizer.add_special_tokens({"mask_token": self.tokenizer.special_tokens_map["eos_token"]})


        with open(f"datasets/wikisql_{self.data_type}_answers.pkl", "rb") as f:
            self.answers = pickle.load(f)

        self.text_input, self.table, self.text_output = self._process_dataset()
        

        self.punctuation_tokens = [".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\",
                     "@", "#", "$", "%", "^", "&", "*", "+", "=", "_", "~", "`"]
        
        self.maskable_words = []
        if self.config.training.training_type == "masked_language_modelling":
            self.tokenized_text = [self.tokenizer.tokenize(x)[:self.config.tokenizer.max_length] for x in self.text_input]
            with open(self.config.data.maskable_words_file, "rb") as f:
                self.maskable_words = pickle.load(f)

    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        pass
        # """
        # Get 0/1 labels for masked tokens with whole word mask proxy
        # """

        # occurrences = [i for i, item in enumerate(input_tokens) if item == self.tokenizer.special_tokens_map["eos_token"]]   
        # # TODO: Fix the IndexError when occurences is an empty list
        # if len(occurrences) == 0:
        #     return torch.tensor([0 for i in range(len(input_tokens))]), -1
        
        # input_tokens_copy = copy.deepcopy(input_tokens)[occurrences[0]+1:]

        # cand_indexes = []
        # for i, token in enumerate(input_tokens_copy):
        #     if token == self.tokenizer.special_tokens_map["bos_token"] or token == self.tokenizer.special_tokens_map["eos_token"] or token in self.punctuation_tokens:
        #         continue

        #     if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
        #         cand_indexes[-1].append(i)
        #     else:
        #         cand_indexes.append([i])       

        # maskable_token_idx = []

        # for i, word in enumerate(cand_indexes):
        #     word = "".join([input_tokens_copy[t] for t in cand_indexes[i]])
        #     word = word.replace("Ġ", "").replace("Â", "")

        #     prev_word = None
        #     if i != 0:
        #         prev_word = "".join([input_tokens_copy[t] for t in cand_indexes[i-1]])
        #         prev_word = prev_word.replace("Ġ", "").replace("Â", "")


        #     if is_maskable(word, self.maskable_words, prev_word = prev_word) and random.random() > 0.5:
        #         maskable_token_idx.extend(cand_indexes[i])

        # maskable_token_idx = [token_idx + occurrences[0] for token_idx in maskable_token_idx]

        # mask_labels = [1 if i in maskable_token_idx else 0 for i in range(len(input_tokens))]
        # return torch.tensor(mask_labels), occurrences[0] + 1



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


    def _soft_decompose_table(self, s1, s2):
        
        # Step 1: Parsing the Table        
        table_string = s2
        table_rows = table_string.split("[ROW]")[1:]  # Skip the header row
        table_columns = table_string.split("[ROW]")[0]

        table_columns_list = [str(col).lower() for col in table_columns.replace("[HEADER]", "").strip().split("|")]
        table_columns_list = [name.strip() for name in table_columns_list]
    
        question = s1
        combined_text = question + ' ' + table_string

        combined_tokens = self.soft_decomposition_tokenizer.tokenize(combined_text, add_special_tokens = True, padding = "max_length", truncation = True)

        combined_input = self.soft_decomposition_tokenizer(combined_text, padding="max_length", truncation=True, return_tensors="pt")
        # combined_input = {"input_ids": combined_input["input_ids"].to("cuda:0"), "attention_mask": combined_input["attention_mask"].to("cuda:0")}
        # combined_input["input_ids"] = combined_input["input_ids"].to("cuda:0")
        # combined_input["attention_mask"] = combined_input["attention_mask"].to("cuda:0")

        with torch.no_grad():
            combined_outputs = self.soft_decomposition_model(**combined_input)

        # Step 3: Masking Question Tokens and Computing Relevance Scores
        question_tokens = self.soft_decomposition_tokenizer.tokenize(question)
        # column_tokens = self.soft_decomposition_tokenizer.tokenize(table_columns)

        relevance_scores = []
        table_embedding = combined_outputs.last_hidden_state[0, len(question_tokens) + 1:].detach().cpu().numpy()


        for masked_token in question_tokens:
            masked_input_ids = combined_input["input_ids"].clone()
            masked_input_ids[0, combined_tokens.index(masked_token)] = self.soft_decomposition_tokenizer.mask_token_id
            masked_input = {"input_ids": masked_input_ids, "attention_mask": combined_input["attention_mask"]}
            with torch.no_grad():
                masked_outputs = self.soft_decomposition_model(**masked_input)

            # Step 4: Cosine Similarity
            masked_embedding = masked_outputs.last_hidden_state[0, len(question_tokens) + 1:].detach().numpy()
            similarity = 1 - cosine_similarity(table_embedding, masked_embedding)
            relevance_scores.append(np.diag(similarity))

            del masked_outputs
            del masked_input_ids
            del masked_input

        relevance_scores = np.mean(np.array(relevance_scores), axis = 0)

        # Step 5: Relevance Score Calculation
        # cell_relevance_scores = []
        cell_relevance_scores = np.zeros((len(table_rows), len(table_columns_list)))

        for i, row in enumerate(table_rows):
            columns = row.split("|")
            # row_scores = []
            for j, cell in enumerate(columns):
                tokenized_cell = self.soft_decomposition_tokenizer.tokenize(cell.strip())
                if len(tokenized_cell) == 0:
                    cell_scores = [0]
                    
                else:
                    cell_scores = [relevance_scores[combined_tokens.index(token) - (len(question_tokens) + 1)] \
                                if token in combined_tokens and (combined_tokens.index(token) - (len(question_tokens) + 1)) < relevance_scores.shape[0] \
                                    else 0 for token in tokenized_cell]
                    
                # row_scores.append(np.mean(cell_scores))
                cell_relevance_scores[i, j] = np.mean(cell_scores)

            # cell_relevance_scores.append(row_scores)

        cell_relevance_scores = 1 / (1 + np.exp(-(0.5 - np.array(cell_relevance_scores) / 0.07)))

        # Pick the top k relevant columns and rows using the cell relevance scores
        flattened_relevance_scores = cell_relevance_scores.flatten()
        topk = min(self.config.data.topk_cells, flattened_relevance_scores.shape[0])

        # if self.config.data.topk_cells > flattened_relevance_scores.shape[0]:
        #     topk = flattened_relevance_scores.shape[0]
        # else:
        #     topk = self.config.data.topk_cells
        top_k_indices = np.argsort(flattened_relevance_scores)[-topk:]

        # Get the row and column indices of the top k cells
        row_indices, col_indices = np.unravel_index(top_k_indices, cell_relevance_scores.shape)

        # Get the corresponding column names and row indices
        top_k_column_names = [table_columns_list[col_idx] for col_idx in col_indices]
        top_k_row_indices = row_indices.tolist()

        return list(set(top_k_row_indices)), list(set(top_k_column_names))


    def expand_numbers(self, strings):
        # Define lookup tables for various place values
        units = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        scales = ['', 'thousand', 'lakh', 'crore']

        expanded_strings = []
        for string in strings:
            match = re.search(r'\d{1,10}(?:,\d{3})*|\d+', string)  # Regular expression to match numbers
            if match:
                number = match.group()
                number = number.replace(',', '')  # Remove commas if present
                number = int(number)
                if number == 0:
                    expanded_string = 'zero'
                else:
                    num_parts = []
                    scale_count = 0
                    while number > 0:
                        num_part = number % 1000  # Consider the last 3 digits
                        if num_part > 0:
                            num_part_words = []
                            hundreds_digit = num_part // 100
                            if hundreds_digit > 0:
                                num_part_words.append(units[hundreds_digit])
                                num_part_words.append('hundred')
                            tens_digit = (num_part // 10) % 10
                            units_digit = num_part % 10
                            if tens_digit == 1:
                                num_part_words.append(teens[units_digit])
                            else:
                                if tens_digit > 1:
                                    num_part_words.append(tens[tens_digit])
                                if units_digit > 0:
                                    num_part_words.append(units[units_digit])
                            num_parts.append(' '.join(num_part_words) + ' ' + scales[scale_count])
                        number //= 1000
                        scale_count += 1
                    num_parts.reverse()
                    expanded_string = ' '.join(num_parts)
                expanded_string = string.replace(match.group(), expanded_string)  # Replace number in string
                expanded_strings.append(expanded_string)
            else:
                expanded_strings.append(string)
        return expanded_strings


    def _process_one_sample(self, data, idx = None):

        question = data["question"]
        table_column_names = data["table"]["header"]
        table_content_values = data["table"]["rows"]

        answer = self.answers[idx]
        # answer_list = answers = [str(a).lower() for a in data["answers"]]
        answer_list = None
        answer = f", ".join(answer).lower()


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
            return question, table, answer

    

        # table = None

        # table = self.soft_decomposed_table[idx]

        # if self.config.tokenizer.special_table_tok:
            
        #     if self.soft_decomposed_table is not None:
        #         if self.config.training.training_type != "table_decomposition":
        #             table = self.soft_decomposed_table[idx]
        #         else:
        #             table_output = self.soft_decomposed_table[idx]
            
        #     else:
        #         table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})

        #         if self.config.data.decompose_table:
        #             if self.config.model.soft_decomposition_model is not None:
        #                 table_string = "[HEADER] " + " | ".join(table_column_names)
        #                 for row in table_content_values:
        #                     table_string += " [ROW] " + " | ".join(row)
        #                 relevant_rows, relevant_columns = self._soft_decompose_table(question, table_string)

        #                 # print(relevant_rows)
        #                 # print(relevant_columns)
                        
        #             else:
        #                 relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                    
        #             if self.config.training.training_type != "table_decomposition":
                        
        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table = table[relevant_columns].iloc[relevant_rows]
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table = table[relevant_columns]
        #             else:
        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table_output = table_output[relevant_columns].iloc[relevant_rows]
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table_output = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table_output = table[relevant_columns]
            
        # else:
            
        #     if self.soft_decomposed_table is not None:
        #         if self.config.training.training_type != "table_decomposition":
        #             table = self.soft_decomposed_table[idx]
        #         else:
        #             table_output = self.soft_decomposed_table[idx]

        #     else:
        #         if self.config.data.decompose_table:
        #             table = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        #             if self.config.model.soft_decomposition_model is not None:
        #                 table_string = "[HEADER] " + " | ".join(table_column_names)
        #                 for row in table_content_values:
        #                     table_string += " [ROW] " + " | ".join(row)
        #                 relevant_rows, relevant_columns = self._soft_decompose_table(question, table_string)
                        
        #             else:
        #                 relevant_rows, relevant_columns = self._decompose_table(question, answer_list, table)
                    
        #             if self.config.training.training_type != "table_decomposition":

        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table = table[relevant_columns].iloc[relevant_rows]
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table = table[relevant_columns]

        #                 table_column_names = table.column.tolist()
        #                 table_content_values = table.values.tolist()

        #             else:
        #                 if self.config.model.soft_decomposition_model is not None:
        #                     table_output = table[relevant_columns].iloc[relevant_rows]
                        
        #                 else:
        #                     if len(relevant_rows) > 0:
        #                         table_output = table.iloc[relevant_rows]
                            
        #                     elif len(relevant_columns) > 0:
        #                         table_output = table[relevant_columns]


        #     table = "[HEADER] " + " | ".join(table_column_names)
        #     for row in table_content_values:
        #         table += " [ROW] " + " | ".join(row) 

        #     if self.config.training.training_type == "table_decomposition":
        #         table_column_names_output = table_output.columns.tolist()
        #         table_content_values_output = table_output.values.tolist()

        #         table_output = "[HEADER] " + " | ".join(table_column_names_output)
        #         for row in table_content_values_output:
        #             table_output += " [ROW] " + " | ".join(row)

        # # if self.config.model.soft_decomposition_model is not None:
        # #     if self.config.tokenizer.special_table_tok:
        # #         raise NotImplementedError
            
        # #     cell_relevance_score = self._get_soft_decomposition_relevance_score(question, table)
        # #     return question, table, answer, cell_relevance_score

        # # else:

        # return question, table, answer

        # if self.config.training.training_type == "table_decomposition":
        #     return question, table, table_output
        # else:
        #     return question, table, answer
    

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
                  or self.config.training.training_type == "table_question_answering" or self.config.training.training_type == "table_decomposition":
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



with open("configs/wiki_tq_reasoning/t5.json", "rb") as f:
    config = json.load(f)
config = process_config(config)

config.training.training_type = "table_question_answering"

dataset = load_dataset("wikisql")

train_dataset = WikiSQLDataset(dataset = dataset, config = config, data_type = "test")

model = T5ModelForTableReasoning(config)
model.load_state_dict(torch.load("logs/table_question_reasoning_flan_t5_xl_reason_with_answer_rerun/checkpoints/epoch=50.pt", map_location = "cpu"))

model.to("cuda:1")
train_dataloader = DataLoader(train_dataset, batch_size = 24, shuffle = False, num_workers = config.system.num_workers)

reason_generations = []

for i, batch in tqdm(enumerate(train_dataloader), position = 0, leave = True, total = len(train_dataloader)):

    input_ids, attention_mask, _, _, labels = batch
    predicted_ids = model.model.generate(input_ids = input_ids.to("cuda:1"), attention_mask = attention_mask.to("cuda:1"), 
                                     max_new_tokens = config.tokenizer.output_max_length, num_beams = 3, early_stopping = True).detach().cpu()

    batch_predicted_reason = train_dataset.tokenizer.batch_decode(predicted_ids, skip_special_tokens = True)

    reason_generations.extend(batch_predicted_reason)


with open("datasets/test_wikisql_reason_without_answer_flant5.pkl", "wb") as f:
    pickle.dump(reason_generations, f)