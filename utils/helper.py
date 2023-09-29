
import os
import re
import sys
import argparse
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from codecs import open
from math import isnan, isinf
from easydict import EasyDict
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod

from data import (
        SciGenDataset, 
        TabFactDataset, 
        ToTToDataset, 
        WikiTQDataset,
        SequentialQADataset,
        FetaQADataset,
        WikiTQWithReasonDataset,
        WikiTQWithReasonAsInputDataset,
        WikiTQWithReasonAsInputBootstrapDataset,
        WikiTQWithReasonAsOutputDataset,
        ToTToCellHighlightingDataset,
        WikiTQHighlightedCellsDataset,
        SequentialQAHighlightedCellsDataset,
        FetaQAHighlightedCellsDataset,
        WikiSQLDataset,
        WikiSQLHighlightedCellsDataset,
        WikiSQLLogicalFormDataset,
        WikiSQLHighlightedCellsLogicalFormDataset,
        WikiSQLWithReasonAsInputDataset,
        FetaQAWithReasonAsInputDataset
    )

from src import (
        BartModelForMaskedLM, 
        BartModelForConditionalGeneration, 
        BartModelForSequenceClassification, 
        BartModelForGenerativeQuestionAnswering,
        EEDBartModelForGenerativeQuestionAnswering,
        DollyModelForConditionalGeneration, 
        T5ModelForConditionalGeneration, 
        T5ModelGenerativeQuestionAnswering,
        GPT2ModelForConditionalGeneration, 
        GPT2ModelForMaskedLM,
        MPTModelForSequenceClassification,
        MPTModelForGenerativeQuestionAnswering,
        GPT2ModelForGenerativeQuestionAnswering,
        EEDBartModelForSequenceClassification,
        CluBartModelForGenerativeQuestionAnswering,
        CluBartModelForSequenceClassification,
        DebertaModelForSequenceClassification,
        CluDebertaModelForSequenceClassification,
        BartModelForTableReasoning,
        T5ModelForTableReasoning,
        ReasoningBartModelForGenerativeQuestionAnswering,
        CluReasoningBartModelForGenerativeQuestionAnswering,
        T5ModelForTableCellHighlighting,
        HighlightedCluBartModelForGenerativeQuestionAnswering,
        BartModelForLogicalFormGeneration,
        HighlightedCluBartModelForLogicalFormGeneration
    )

def process_config(config: dict, args = None):

    if args is not None:
        args_dict = vars(args)
        merged_dict = {**args_dict, **config}
    else:
        merged_dict = config

    merged_config = EasyDict(merged_dict)
    return merged_config

def prepare_dataloaders(dataset, config):

    if config.data.name == "scigen":
        train_dataset = SciGenDataset(dataset, config, "train")
        validation_dataset = SciGenDataset(dataset, config, "validation")
        test_dataset = SciGenDataset(dataset, config, "test")

    elif config.data.name == "tabfact":
        train_dataset = TabFactDataset(dataset, config, "train")
        validation_dataset = TabFactDataset(dataset, config, "validation")
        test_dataset = TabFactDataset(dataset, config, "test")

    elif config.data.name == "totto":
        if config.training.training_type == "table_cell_highlighting":
            train_dataset = ToTToCellHighlightingDataset(dataset, config, "train")
            test_dataset = ToTToCellHighlightingDataset(dataset, config, "test")
            validation_dataset = ToTToCellHighlightingDataset(dataset, config, "validation")

        else:
            train_dataset = ToTToDataset(dataset, config, "train")
            validation_dataset = ToTToDataset(dataset, config, "validation")
            test_dataset = ToTToDataset(dataset, config, "test")

    elif config.data.name == "wikitq":
        if config.data.use_reason_in_input:
            if config.data.bootstrap:
                train_dataset = WikiTQWithReasonAsInputBootstrapDataset(dataset, config, "train")
                test_dataset = WikiTQWithReasonAsInputBootstrapDataset(dataset, config, "test")
                validation_dataset = WikiTQWithReasonAsInputBootstrapDataset(dataset, config, "validation")
            else:
                train_dataset = WikiTQWithReasonAsInputDataset(dataset, config, "train")
                test_dataset = WikiTQWithReasonAsInputDataset(dataset, config, "test")
                validation_dataset = WikiTQWithReasonAsInputDataset(dataset, config, "validation")

        elif config.data.use_highlighted_cells:
            train_dataset = WikiTQHighlightedCellsDataset(dataset, config, "train")
            test_dataset = WikiTQHighlightedCellsDataset(dataset, config, "test")
            validation_dataset = WikiTQHighlightedCellsDataset(dataset, config, "validation")

        else:
            train_dataset = WikiTQDataset(dataset, config, "train")
            test_dataset = WikiTQDataset(dataset, config, "test")
            validation_dataset = WikiTQDataset(dataset, config, "validation")

    elif config.data.name == "sequentialqa":
        if config.data.use_highlighted_cells:
            train_dataset = SequentialQAHighlightedCellsDataset(dataset, config, "train")
            test_dataset = SequentialQAHighlightedCellsDataset(dataset, config, "test")
            validation_dataset = SequentialQAHighlightedCellsDataset(dataset, config, "validation")
        else:
            train_dataset = SequentialQADataset(dataset, config, "train")
            test_dataset = SequentialQADataset(dataset, config, "test")
            validation_dataset = SequentialQADataset(dataset, config, "validation")

    elif config.data.name == "fetaqa":

        if config.data.use_reason_in_input: 
            train_dataset = FetaQAWithReasonAsInputDataset(dataset, config, "train")
            test_dataset = FetaQAWithReasonAsInputDataset(dataset, config, "test")
            validation_dataset = FetaQAWithReasonAsInputDataset(dataset, config, "validation")
        else:
            if config.data.use_highlighted_cells:
                train_dataset = FetaQAHighlightedCellsDataset(dataset, config, "train")
                test_dataset = FetaQAHighlightedCellsDataset(dataset, config, "test")
                validation_dataset = FetaQAHighlightedCellsDataset(dataset, config, "validation")
            else:
                train_dataset = FetaQADataset(dataset, config, "train")
                test_dataset = FetaQADataset(dataset, config, "test")
                validation_dataset = FetaQADataset(dataset, config, "validation")

    elif config.data.name == "wikitq_with_reason":
        train_dataset = WikiTQWithReasonDataset(dataset, config, "train")
        test_dataset = WikiTQWithReasonDataset(dataset, config, "test")
        validation_dataset = WikiTQWithReasonDataset(dataset, config, "validation")

    elif config.data.name == "wikitq_with_reason_and_answer":
        train_dataset = WikiTQWithReasonAsOutputDataset(dataset, config, "train")
        test_dataset = WikiTQWithReasonAsOutputDataset(dataset, config, "test")
        validation_dataset = WikiTQWithReasonAsOutputDataset(dataset, config, "validation")

    elif config.data.name == "wikisql":

        if config.training.training_type == "table_logic_generation":
            
            if config.data.use_highlighted_cells:
                train_dataset = WikiSQLHighlightedCellsLogicalFormDataset(dataset, config, "train")
                test_dataset = WikiSQLHighlightedCellsLogicalFormDataset(dataset, config, "test")
                validation_dataset = WikiSQLHighlightedCellsLogicalFormDataset(dataset, config, "validation")
            else:
                train_dataset = WikiSQLLogicalFormDataset(dataset, config, "train")
                test_dataset = WikiSQLLogicalFormDataset(dataset, config, "test")
                validation_dataset = WikiSQLLogicalFormDataset(dataset, config, "validation")
        else:

            if config.data.use_reason_in_input: 
                train_dataset = WikiSQLWithReasonAsInputDataset(dataset, config, "train")
                test_dataset = WikiSQLWithReasonAsInputDataset(dataset, config, "test")
                validation_dataset = WikiSQLWithReasonAsInputDataset(dataset, config, "validation")
                
            else:
                if config.data.use_highlighted_cells:
                    train_dataset = WikiSQLHighlightedCellsDataset(dataset, config, "train")
                    test_dataset = WikiSQLHighlightedCellsDataset(dataset, config, "test")
                    validation_dataset = WikiSQLHighlightedCellsDataset(dataset, config, "validation")
                else:
                    train_dataset = WikiSQLDataset(dataset, config, "train")
                    test_dataset = WikiSQLDataset(dataset, config, "test")
                    validation_dataset = WikiSQLDataset(dataset, config, "validation")

    train_dataloader = DataLoader(train_dataset, batch_size = config.training.train_batch_size, shuffle = True, num_workers = config.system.num_workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size = config.training.train_batch_size, shuffle = False, num_workers = config.system.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size = config.training.test_batch_size, shuffle = False, num_workers = config.system.num_workers)

    return train_dataloader, validation_dataloader, test_dataloader, test_dataset.tokenizer


def filter_cell(cell):

    num_list = re.findall("\d+\.\d+", cell)
    if len(num_list) == 0:
        return cell
    else:
        return float(num_list[0])


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
            new_col = np.round(np.array(table_copy[numeric_cols[col_idx_1]]) * np.array(table_copy[numeric_cols[col_idx_2]]), 3)
            table["new"] = new_col
            desc = f"The new column is derived by multiplying {numeric_cols[col_idx_1]} and {numeric_cols[col_idx_2]}"
            dependency.append(desc)

        else: 
            table["new"] = np.random.randint(0, 2, size = len(table))
            desc = "The new column has no correlation with any column"
            dependency.append(str(desc))

        table["new"] = table["new"].astype(str)

        table_column_names.append("new")
        table_column_names_synthetic.append(str(table_column_names))

        table_content_values_synthetic.append(str([[table[col][i] for col in table_column_names] for i in range(len(table))]))

    dataset[data_type] = dataset[data_type].remove_columns("table_column_names")
    dataset[data_type] = dataset[data_type].remove_columns("table_content_values")

    dataset[data_type] = dataset[data_type].add_column("table_content_values", table_content_values_synthetic)
    dataset[data_type] = dataset[data_type].add_column("table_column_names", table_column_names_synthetic)
    dataset[data_type] = dataset[data_type].add_column("dependency", dependency)

    return dataset

# Personal logger (never used in code)
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

def prepare_models(config):

    if config.training.training_type == "sequence_classification":

        if config.model.model_name == "tapex":
            if config.model.num_encoders == 2:
                if config.model.cluster_encodings: 
                    model = CluBartModelForSequenceClassification(config)
                else:
                    model = EEDBartModelForSequenceClassification(config)
                    
                # for name, param in model.named_parameters():
                #     if "shared" in name or "decomposer" in name or "token_classifier" in name:
                #         param.requires_grad = True
                #     else:
                #         param.requires_grad = False
                # for name, param in model.named_parameters():
                    # if not param.requires_grad:
                    #     print(name, param.requires_grad)

            else:
                model = BartModelForSequenceClassification(config)

        elif config.model.model_name == "mpt":
            model = MPTModelForSequenceClassification(config)
            # flag = False
            # for name, param in model.named_parameters():
            #     if "26" not in name and not flag:
            #         param.requires_grad = flag
            #     else:
            #         flag = True
            #         break

        elif config.model.model_name == "pasta":
            if config.model.cluster_encodings:
                model = CluDebertaModelForSequenceClassification(config)
            else:
                model = DebertaModelForSequenceClassification(config)
                

    elif config.training.training_type == "description_generation" or config.training.training_type == "column_reasoning":

        if config.model.model_name == "tapex":
            model = BartModelForConditionalGeneration(config)

        elif config.model.model_name == "dolly":
            model = DollyModelForConditionalGeneration(config)
            for i, (name, param) in enumerate(model.named_parameters()):
                if i < 250:
                    param.requires_grad = False

        elif config.model.model_name == "t5":
            model = T5ModelForConditionalGeneration(config)

        elif config.model.model_name == "gpt2":
            model = GPT2ModelForConditionalGeneration(config)

    elif config.training.training_type == "masked_language_modelling":

        if config.model.model_name == "tapex":
            model = BartModelForMaskedLM(config)
        
        elif config.model.model_name == "gpt2":
            model = GPT2ModelForMaskedLM(config)

    elif config.training.training_type == "table_question_answering" or config.training.training_type == "descriptive_table_question_answering":

        if config.model.model_name == "tapex":
            if config.model.num_encoders == 2:
                if config.model.cluster_encodings:
                    if config.data.use_highlighted_cells:
                        model = HighlightedCluBartModelForGenerativeQuestionAnswering(config)
                    else:
                        model = CluBartModelForGenerativeQuestionAnswering(config)
                else:
                    model = EEDBartModelForGenerativeQuestionAnswering(config)

            else:
                model = BartModelForGenerativeQuestionAnswering(config)

        elif config.model.model_name == "mpt":
            model = MPTModelForGenerativeQuestionAnswering(config)
            # flag = False
            # for name, param in model.named_parameters():
            #     if "27" not in name and not flag:
            #         param.requires_grad = flag
            #     else:
            #         flag = True
            #         break

        elif config.model.model_name == "gpt2":
            model = GPT2ModelForGenerativeQuestionAnswering(config)

        elif config.model.model_name == "t5":
            model = T5ModelGenerativeQuestionAnswering(config)

            # flag = False
            # for name, param in model.named_parameters():
            #     if "18" not in name and "decoder" not in name and not flag:
            #         param.requires_grad = flag
            #     else:
            #         flag = True
            #         break

    elif config.training.training_type == "table_decomposition":

        if config.model.model_name == "tapex":
            model = BartModelForGenerativeQuestionAnswering(config)

    elif config.training.training_type == "table_reasoning":

        if config.model.model_name == "tapex":
            model = BartModelForTableReasoning(config)
        
        elif config.model.model_name == "t5":
            model = T5ModelForTableReasoning(config)


    elif config.training.training_type == "table_question_answering_with_reason" or config.training.training_type == "table_question_answering_with_clustering_and_reasoning":
        
        if config.model.model_name == "tapex":
            if config.model.num_encoders == 3:
                model = CluReasoningBartModelForGenerativeQuestionAnswering(config)
            else:
                model = ReasoningBartModelForGenerativeQuestionAnswering(config)


    elif config.training.training_type == "table_question_reasoning_and_answering":

        if config.model.model_name == "tapex":
            if config.model.cluster_encodings:
                model = CluBartModelForGenerativeQuestionAnswering(config)
            else:
                model = BartModelForGenerativeQuestionAnswering(config)


    elif config.training.training_type == "table_cell_highlighting":

        if config.model.model_name == "t5":
            model = T5ModelForTableCellHighlighting(config)

    elif config.training.training_type == "table_logic_generation":
        
        if config.model.model_name == "tapex":
            if config.model.cluster_encodings:
                model = HighlightedCluBartModelForLogicalFormGeneration(config)
            else:
                model = BartModelForLogicalFormGeneration(config)

    return model


def normalize(x):

    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')

    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)

    while True:
        
        old_x = x

        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        
        if x == old_x:
            break
    
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    
    return x


class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):

        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        
        self._year = year
        self._month = month
        self._day = day
        
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))

    __repr__ = __str__

    def match(self, other):
        
        assert isinstance(other, Value)
        
        if self.normalized == other.normalized:
            return True
        
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """

    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    
    if not corenlp_value:
        corenlp_value = original_string
    
    # Number?
    amount = NumberValue.parse(corenlp_value)
    
    if amount is not None:
        return NumberValue(amount, original_string)
    
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                        in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """

    # Check size
    if len(target_values) != len(predicted_values):
        return False
    
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    
    return True


def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')


def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]