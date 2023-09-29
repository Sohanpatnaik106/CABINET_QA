
import os
os.chdir("../")

import re
import json
import torch
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset


from utils import (
        process_config, 
        set_seed,
        prepare_dataloaders
    )

from data import (
        SciGenDataset, 
        TabFactDataset, 
        ToTToDataset, 
        WikiTQDataset
    )

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def create_synthetic_column(dataset, data_type = "train"):

    dependency = []
    table_content_values_synthetic = []
    table_column_names_synthetic = []

    for i, data in tqdm(enumerate(dataset[data_type]), position = 0, leave = True, total = len(dataset[data_type])):

        table_column_names = eval(data["table_column_names"])
        table_content_values = eval(data["table_content_values"])
        
        table = pd.DataFrame.from_dict({col: [table_content_values[j][i] for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})
        table_copy = pd.DataFrame.from_dict({col: [filter_cell(table_content_values[j][i]) for j in range(len(table_content_values))] for i, col in enumerate(table_column_names)})

        numeric_cols = table_copy.select_dtypes('number').columns

        if len(numeric_cols) == 3 or len(numeric_cols) == 2:
            weights = np.random.randn(len(numeric_cols))
            new_col = np.round(np.sum(np.array(table_copy[numeric_cols]) * (weights / np.sum(weights)), axis = 1), 3)
            table["new"] = new_col
            if len(numeric_cols) == 3:
                desc = f"The new column is derived using a weighted combination of {numeric_cols[0]}, {numeric_cols[1]} and {numeric_cols[2]}"
            else:
                desc = f"The new column is derived using a weighted combination of {numeric_cols[0]} and {numeric_cols[1]}"
            dependency.append(desc)

        elif len(numeric_cols) == 1:
            mean = np.mean(table_copy[numeric_cols[0]])
            table["new"] = [1 if table_copy[numeric_cols[0]][i] > mean else 0 for i in range(len(table_copy))]
            desc = f"The new column has a value 1 if the value of {numeric_cols[0]} is greater than the mean of all values in that column"
            dependency.append(desc)

        elif len(numeric_cols) == 4:
            col_idx = np.random.randint(len(numeric_cols))
            median = np.median(table_copy[numeric_cols[col_idx]])
            table["new"] = [1 if table_copy[numeric_cols[col_idx]][i] > median else 0 for i in range(len(table_copy))]
            desc = f"The new column has a value 1 if the value of {numeric_cols[col_idx]} is greater than the median of all values in that column"
            dependency.append(desc)

        elif len(numeric_cols) > 4:
            col_idx_1, col_idx_2 = np.random.randint(len(numeric_cols)), np.random.randint(len(numeric_cols))
            new_col = np.array(table_copy[numeric_cols[col_idx_1]]) * np.array(table_copy[numeric_cols[col_idx_2]])
            table["new"] = new_col
            desc = f"The new column is derived by multiplying {numeric_cols[col_idx_1]} and {numeric_cols[col_idx_2]}"
            dependency.append(desc)

        else: 
            table["new"] = np.random.randint(0, 2, size = len(table))
            desc = "The new column has no correlation with any column"
            dependency.append(desc)

        table["new"] = table["new"].astype(str)

        table_column_names.append("new")
        table_column_names_synthetic.append(str(table_column_names))

        table_content_values_synthetic.append([[table[col][i] for col in table_column_names] for i in range(len(table))])

    dataset[data_type] = dataset[data_type].remove_columns("table_column_names")
    dataset[data_type] = dataset[data_type].remove_columns("table_content_values")

    dataset[data_type] = dataset[data_type].add_column("table_content_values", table_content_values_synthetic)
    dataset[data_type] = dataset[data_type].add_column("table_column_names", table_column_names_synthetic)
    dataset[data_type] = dataset[data_type].add_column("dependency", dependency)

    return dataset


def filter_cell(cell):

    num_list = re.findall("\d+\.\d+", cell)
    if len(num_list) == 0:
        return cell
    else:
        return float(num_list[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/t5_baseline_desc_gen.json", type = str, help = "Path to experiment configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)
    set_seed(config.seed)

    dataset = load_dataset("kasnerz/scigen")

    # NOTE: Uncomment the following lines if new dataset with synthetic column is 
    # required to be created and dumped as hugginface datasets
    
    # dataset = create_synthetic_column(dataset, "train")
    # dataset = create_synthetic_column(dataset, "validation")
    # dataset = create_synthetic_column(dataset, "test")
    # dataset.save_to_disk("datasets/scigen")