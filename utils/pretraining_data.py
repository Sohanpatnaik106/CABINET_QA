
import os
os.chdir("../")

import torch
import torch.nn as nn
import pandas as pd

import json

from tqdm import tqdm



import os
import copy
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from joblib import Parallel, delayed


def _add_adjusted_col_offsets(table):
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


def _get_heuristic_col_headers(adjusted_table, row_index, col_index):
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


def get_totto_full_table(table, cell_indices, table_page_title = None, table_section_title = None):

    """Verbalize full table and return a string."""
    table_str = "Start of a new table with repetition of column names in between for your reference\n"
    if table_page_title:
        table_str += "<page_title> " + table_page_title + " </page_title> "
    if table_section_title:
        table_str += "<section_title> " + table_section_title + " </section_title> "

    adjusted_table = _add_adjusted_col_offsets(table)

    col_headers = []
    for r_index, row in enumerate(table):
        row_str = "<row> "
        for c_index, col in enumerate(row):
            col_header = _get_heuristic_col_headers(adjusted_table, r_index, c_index)
            
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



def get_data_totto(dataset, split = "train"):


    processed_data = Parallel(n_jobs = -1)(
        delayed(get_totto_full_table)(data["table"], data["highlighted_cells"]) for i, data in tqdm(enumerate(dataset), position = 0, leave = True, total = len(dataset))
    )

    input_text = []
    table_list = []
    output_text = []

    for i, data in tqdm(enumerate(processed_data), position = 0, leave = True, total = len(processed_data)):
        # id = dataset[i]["totto_id"]
        # table_page_title = dataset[i]["table_page_title"]
        # table_webpage_url = dataset[i]["table_webpage_url"]
        # table_section_title = dataset[i]["table_section_title"]
        # table_section_text = dataset[i]["table_section_text"]
        # table = dataset[i]["table"]
        # cell_indices = dataset[i]["highlighted_cells"]
        target = dataset[i]["target"]

        table = data[0]
        highlighted_cells = data[1]

        input_text.append(f"Find the highlighted cells from table for given statement. Statement: {target}")
        table_list.append(table)
        output_text.append(",".join(highlighted_cells))

        # verbalised_dataset[i] = {"id": id,
        #                         "table_page_title": table_page_title,
        #                         "table_webpage_url": table_webpage_url,
        #                         "table_section_title": table_section_title,
        #                         "table_section_text": table_section_text,
        #                         "actual_table": table,
        #                         "highlighted_cell_indices": cell_indices,
        #                         "highlighted_cells": highlighted_cells,
        #                         "verbalised_table": verbalised_table}


    return input_text, table_list, output_text

def load_a_table(filename):
    table_contents = []
    table_df = pd.read_csv(filename, sep='#').astype(str)

    table_contents.append(table_df.columns.tolist()) #header
    table_contents.extend(table_df.values.tolist()) #rows

    return table_contents


def load_the_tables(dataset_path = None):
    table_path = os.path.join(dataset_path, 'data/all_csv')
        
    print(f"load tables from {table_path}")
    tables = {}
    for filename in tqdm(os.listdir(table_path), position = 0, leave = True):
        tables[filename] = load_a_table(os.path.join(table_path, filename))
    return tables


def load_all_the_data_pasta(dataset_path = None):
    all_data = {}
    for filename in tqdm(os.listdir(os.path.join(dataset_path, "raw_data/")), position = 0, leave = True):
        infos = eval(open(os.path.join(dataset_path, "raw_data/", filename)).read())
        all_data.update(infos)

    print(len(all_data))
    return all_data


def get_data_pasta(split = "train", dataset_path = None, all_data = None, tables = None):
    if split == 'train':
        filename = os.path.join(dataset_path, 'data/train_id.json')
        cache_filename = os.path.join(dataset_path, 'train')
    elif split == 'val':
        filename = os.path.join(dataset_path, 'data/val_id.json')
        cache_filename = os.path.join(dataset_path, 'val')
    else:
        assert 1 == 2, "which should be in train/val"

    table_ids = eval(open(filename).read())
    # samples = []
    input_text = []
    table_list = []
    output_text = []

    for tab_id in tqdm(table_ids, position = 0, leave = True, total = len(table_ids)):
        if tab_id not in all_data.keys():
            continue
        tab_data = all_data[tab_id]
        sentences, clozes = tab_data
        table = tables[tab_id]
        table_dict = {}
        table_dict["header"] = table[0]
        table_dict["rows"] = table[1:]

        for sentence, cloze in zip(sentences, clozes):
            input_text.append(f"Replace [MASK] token with suitable word. Statement: {cloze}")
            table_list.append(table_dict)
            output_text.append(sentence)

    print(f'{split} sample num = {len(input_text)}')
    return input_text, table_list, output_text



def get_data_reastap(split = "train", dataset_path = None):
    with open(os.path.join(dataset_path, "train.jsonl"), "r") as f:
        json_list = list(f)

    print(len(json_list))
    input_text = []
    table_list = []
    output_text = []

    for i, json_str in tqdm(enumerate(json_list), position = 0, leave = True, total = len(json_list)):

        result = json.loads(json_str)

        if result["source"] == "tapex":
            input_text.append(f"Execute the SQL query on the corresponding table. SQL Query: {result['question']}")

        elif result["source"] == "synthetic_qa":
            input_text.append(f"Answer the question based on the table. Question: {result['question']}")

        table_list.append(result["table"])
        output_text.append(",".join(result["answers"]))

    return input_text, table_list, output_text


if __name__ == "__main__":


    """
        Multiple datasets are combined to prepare the pre-training dataset    
    """

    # NOTE: Get the data corresponding the PASTA approach
    # pasta_dataset_path = "PASTA/pasta_corpus"
    # pasta_all_data = load_all_the_data_pasta(dataset_path = pasta_dataset_path)
    # tables = load_the_tables(dataset_path = pasta_dataset_path)
    # pasta_input_text, pasta_tables, pasta_output_text = get_data_pasta(split = "train", dataset_path = pasta_dataset_path, all_data = pasta_all_data, tables = tables)

    # NOTE: Get the data corresponding to ReasTAP approach, both synthetic QA and tapex data present
    # reastap_dataset_path = "ReasTAP/pretrain_data"
    # reastap_text_input, reastap_tables, reastap_text_output = get_data_reastap(split = "train", dataset_path = reastap_dataset_path)


    # NOTE: Get the data correspondinf to ToTTo, given statement and table, find highlighted cells
    dataset = load_dataset("GEM/totto")
    train_dataset = dataset["train"]
    totto_text_input, totto_tables, totto_text_output = get_data_totto(dataset = train_dataset, split = "train")


    print(totto_text_input[0])
    print(len(totto_text_input))
    print(totto_tables[0])
    print(len(totto_tables))
    print(totto_text_output[0])
    print(len(totto_text_output))