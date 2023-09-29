
import os
os.chdir("../")

import json
from src import T5ModelForTableCellHighlighting
from datasets import load_dataset
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from utils import process_config

from tqdm import tqdm

dataset = load_dataset("wikisql")

with open("configs/cell_highlighting/t5.json", "rb") as f:
    config = json.load(f)
config = process_config(config)

model = T5ModelForTableCellHighlighting(config)
model.load_state_dict(torch.load("logs/table_cell_highlighting_flan_t5_xl_pretrain/checkpoints/epoch=2.pt", map_location="cpu"))

model.to("cuda:0")

with open("datasets/wikisql_reason_without_answer_flant5.pkl", "rb") as f:
    reasons_list = pickle.load(f)


tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)

def predict(idx):

    reason = reasons_list[idx]

    table_column_names = dataset["train"][idx]["table"]["header"]
    table_content_values = dataset["train"][idx]["table"]["rows"]

    table_column_names = [x.lower() for x in table_column_names]

    table = "[HEADER] " + " | ".join(table_column_names)
    for row_id, row in enumerate(table_content_values):
        row = [x.lower() for x in row]
        table += f" [ROW] {row_id}: " + " | ".join(row) 

    tokenized_input = tokenizer(reason, table, add_special_tokens = config.tokenizer.add_special_tokens,
                            padding = config.tokenizer.padding, truncation = config.tokenizer.truncation, 
                            max_length = config.tokenizer.input_max_length, return_tensors = config.tokenizer.return_tensors,
                            return_token_type_ids = config.tokenizer.return_token_type_ids,
                            return_attention_mask = config.tokenizer.return_attention_mask)


    output_ids = model.model.generate(input_ids = tokenized_input["input_ids"].to("cuda:0"), attention_mask = tokenized_input["attention_mask"].to("cuda:0"),
                                      max_new_tokens = config.tokenizer.output_max_length, num_beams = 3, early_stopping = True).squeeze().detach().cpu()

    predicted_cells = tokenizer.decode(output_ids, skip_special_tokens=True)

    return predicted_cells

from tqdm import tqdm
highlighted_cells_list = []

for i in tqdm(range(len(reasons_list)), position=0, leave = True, total = len(reasons_list)):
    x = predict(i).split(", ")
    x = [a.strip() for a in x]

    highlighted_cells_list.append(x)

with open("datasets/wikisql_train_highlighted_cell_flant_t5_reasons.pkl", "wb") as f:
    pickle.dump(highlighted_cells_list, f)


# with open("datasets/seq_qa_train_highlighted_cell_flant_t5_reasons_no_history.pkl", "rb") as f:
#     highlighted_cells_list = pickle.load(f)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large")

train_dataset = dataset["train"]

import pandas as pd
hard_relevance_labels = []

for i in tqdm(range(len(train_dataset)), position = 0, leave = True, total = len(train_dataset)):

    question = " ".join(train_dataset[i]["question"])
    table_column_names = train_dataset[i]["table"]["header"]
    table_content_values = train_dataset[i]["table"]["rows"]
    # question = train_dataset[i]["question"]
    highlighted_cells = highlighted_cells_list[i]

    # table = train_dataset[i]["table"]
    # table_column_names = table["header"]    
    # table_content_values = table["rows"]

    table_df = pd.DataFrame.from_dict({str(col).lower(): [str(table_content_values[j][i]).lower() for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
    
    tokenized_input = tokenizer(table_df, question, add_special_tokens = config.tokenizer.add_special_tokens,
                            padding = config.tokenizer.padding, truncation = config.tokenizer.truncation, 
                            max_length = 960, return_tensors = config.tokenizer.return_tensors,
                            return_token_type_ids = config.tokenizer.return_token_type_ids,
                            return_attention_mask = config.tokenizer.return_attention_mask)
    
    tokenized_highlighted_cells = []
    hard_relevance_label = torch.zeros((tokenized_input["input_ids"].shape[1]))
    for h_cell in highlighted_cells:
        x = tokenizer(answer = h_cell, add_special_tokens = False,
                            return_tensors = config.tokenizer.return_tensors,
                            return_attention_mask = config.tokenizer.return_attention_mask)["input_ids"].tolist()
        for ele in x[0]:
            hard_relevance_label[tokenized_input["input_ids"].squeeze() == ele] = 1
        
    hard_relevance_labels.append(hard_relevance_label)

with open("datasets/wikisql_train_highlighted_cell.pkl", "wb") as f:
    pickle.dump(hard_relevance_labels, f)