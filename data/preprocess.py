import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from .utils import is_maskable


def mask_tokens(text, tokenizer, maskable_words = None):

    # Find the indices of the condition words in the tokenized input
    tokenized_inputs = tokenizer.tokenize(text = text)
    
    # Convert the tokenized input to token IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_inputs)

    # Find the indices of the condition words in the tokenized input
    # condition_indices = []
    # for condition_word in get_maskable_words():
    condition_indices = [i for i, token in enumerate(tokenized_inputs) if is_maskable(token)]

    # Mask the tokens corresponding to the condition words
    for index in condition_indices:
        input_ids[index] = tokenizer.mask_token_id

    # Convert the input IDs back to tokens
    masked_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Convert the masked tokens back to a sentence
    masked_sentence = tokenizer.convert_tokens_to_string(masked_tokens)

    return masked_sentence


def mask_tokens_wrapper(text_output, tokenizer = None, config = None):

    if os.path.exists(config.data.maskable_words_file):
        with open(config.data.maskable_words_file, "rb") as f:
            maskable_words = pickle.load(f)

    if os.path.exists(config.data.masked_output_file):
        with open(config.data.masked_output_file, "rb") as f:
            masked_output = pickle.load(f)
    else:
        masked_output = Parallel(n_jobs=-1)(
            delayed(mask_tokens)(
                text, tokenizer, maskable_words
            ) for text in tqdm(text_output, position = 0, leave = True, 
                    total = len(text_output), desc = "Masking")
        )
        with open(config.data.masked_output_file, "wb") as f:
            pickle.dump(masked_output, f)
    
    return tokenizer(answer = masked_output, add_special_tokens = config.tokenizer.add_special_tokens,
                padding = config.tokenizer.padding, truncation = config.tokenizer.truncation, 
                max_length = config.tokenizer.max_length, return_tensors = config.tokenizer.return_tensors)