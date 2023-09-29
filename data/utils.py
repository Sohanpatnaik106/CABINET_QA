
import re
import json
import nltk
import pickle
import random
import argparse
from easydict import EasyDict
from collections import Counter
from nltk.corpus import stopwords
from datasets import load_dataset
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader

import six
import enum
import functools
from tqdm import tqdm
from typing import Iterable, Tuple, Text, Any, Mapping, MutableMapping, List, Optional


def process_config(config: dict, args: argparse.Namespace):

    args_dict = vars(args)
    merged_dict = {**args_dict, **config}
    merged_config = EasyDict(merged_dict)

    return merged_config

def contraction_expansion(word):

    # Dictionary of common contractions

    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'll": "I will",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you've": "you have"
    }

    # Expand contractions
    if word in contractions_dict:
        word = contractions_dict[word]

    return word

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


def preprocess_sentences(sentences):
    # Combine all sentences into a single string
    text = ' '.join(sentences)
    
    text = preprocess_text(text)
    
    return text

def create_word_frequency_dict(words):
    # Count the frequency of each word
    word_counts = Counter(words)
    
    # Sort the words by frequency in descending order
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create the dictionary of words and their frequencies
    word_frequency_dict = {word: count for word, count in sorted_words}
    
    return word_frequency_dict


def construct_count_vocab(config):
    dataset = load_dataset(config.data.data_path)["train"]

    corpus = [data["text"] for data in dataset]
    corpus = preprocess_sentences(corpus)

    return create_word_frequency_dict(corpus)


def is_number(token):
    pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    return bool(re.match(pattern, token))

    
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


## WikiSQL utils
# Functions to obtain the answer text by executing sql statement over a table


def _split_thousands(delimiter, value):
    split = value.split(delimiter)
    return len(split) > 1 and any(map(lambda x: len(x) == 3, split))


def convert_to_float(value):
    """Converts value to a float using a series of increasingly complex heuristics.

    Args:
        value: object that needs to be converted. Allowed types include
          float/int/strings.

    Returns:
        A float interpretation of value.

    Raises:
        ValueError if the float conversion of value fails.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, six.string_types):
        raise ValueError("Argument value is not a string. Can't parse it as float")
    sanitized = value

    try:
        # Example: 1,000.7
        if "." in sanitized and "," in sanitized:
            return float(sanitized.replace(",", ""))
        # 1,000
        if "," in sanitized and _split_thousands(",", sanitized):
            return float(sanitized.replace(",", ""))
        # 5,5556
        if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
            ",", sanitized):
            return float(sanitized.replace(",", "."))
        # 0.0.0.1
        if sanitized.count(".") > 1:
            return float(sanitized.replace(".", ""))
        # 0,0,0,1
        if sanitized.count(",") > 1:
            return float(sanitized.replace(",", ""))
        return float(sanitized)
    except ValueError:
        # Avoid adding the sanitized value in the error message.
        raise ValueError("Unable to convert value to float")


_TABLE_DIR_NAME = 'table_csv'  # Name that the table folder has in SQA.

_TYPE_CONVERTER = {
    'text': lambda x: x,
    'real': convert_to_float,
}

_TOKENIZER = re.compile(r'\w+|[^\w\s]+', re.UNICODE | re.MULTILINE | re.DOTALL)

_DATASETS = ['train', 'test', 'dev']

_NAN = float('nan')


class _Aggregation(enum.Enum):
    """Aggregations as defined by WikiSQL. Indexes match the data."""
    NONE = 0
    MAX = 1
    MIN = 2
    COUNT = 3
    SUM = 4
    AVERAGE = 5


class _Operator(enum.Enum):
    """The boolean operators used by WikiSQL. Indexes match the data."""
    EQUALS = 0
    GREATER = 1
    LESSER = 2


class _Condition:
    """Represents an SQL where clauses (e.g A = "a" or B > 5)."""
    def __init__(self, column, operator, cmp_value):
        self.column = column
        self.operator = operator
        self.cmp_value = cmp_value
    # column: Text
    # operator: _Operator
    # cmp_value: Any


def _parse_value(table, column,
                 cell_value):
    """Convert numeric values to floats and keeps everything else as string."""
    types = table['types']
    return _TYPE_CONVERTER[types[column]](cell_value)


def _parse_table(table):
    """Runs the type converter over the table cells."""
    types = table['types']
    table['real_rows'] = table['rows']
    typed_rows = []
    for row in table['rows']:
        typed_row = []
        for column, cell_value in enumerate(row):
            typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
        typed_rows.append(typed_row)
    table['rows'] = typed_rows


def _compare(operator, src, tgt):
    if operator == _Operator.EQUALS:
        return src == tgt
    elif operator == _Operator.GREATER:
        return src > tgt
    elif operator == _Operator.LESSER:
        return src < tgt
    raise ValueError(f'Unknown operator: {operator}')


def _is_string(x):
    return isinstance(x, str)


def _normalize_for_match(x):
    return [t for t in _TOKENIZER.findall(x.lower())]


def _respect_conditions(table, row,
                        conditions):
    """True if 'row' satisfies all 'conditions'."""
    for cond in conditions:
        table_value = row[cond.column]
        if not _is_string(table_value):
            cmp_value = _parse_value(table, cond.column, cond.cmp_value)
        else:
            cmp_value = str(cond.cmp_value)
        # print("Table value: ", table_value)
        # print("cmp value: ", cmp_value)

        if _is_string(table_value) and _is_string(cmp_value):
            table_value = _normalize_for_match(table_value)
            cmp_value = _normalize_for_match(cmp_value)

        if not isinstance(table_value, type(cmp_value)):
            raise ValueError('Type difference {} != {}'.format(
                type(table_value), type(cmp_value)))

        if not _compare(cond.operator, table_value, cmp_value):
            return False
    return True


def _get_answer_coordinates(
        table,
        example):
    """Retrieves references coordinates by executing SQL."""
    # MAX and MIN are automatically supported by the model.
    # aggregation_op_index = example['sql']['agg']
    aggregation_op_index = example['agg']
    if aggregation_op_index >= 3:
        aggregation_op = _Aggregation(aggregation_op_index)
    else:
        aggregation_op = _Aggregation.NONE
    
    # target_column = example['sql']['sel']
    target_column = example['sel']
    conditions = [
        _Condition(column, _Operator(operator), cmp_value)
        # for column, operator, cmp_value in example['sql']['conds']
        for column, operator, cmp_value in zip(example["conds"]["column_index"], example["conds"]["operator_index"], example["conds"]["condition"])
    ]

    indices = []
    for row in range(len(table['rows'])):
        if _respect_conditions(table, table['rows'][row], conditions):
            indices.append((row, target_column))
    if not indices:
        return [], aggregation_op

    if len(indices) == 1:
        return indices, aggregation_op

    # Parsing of MIN/MAX.
    if aggregation_op_index in (1, 2):
        operators = {2: min, 1: max}
        values = [
            (table['rows'][i][j], index) for index, (i, j) in enumerate(indices)
        ]
        # reduced = functools.reduce(operators[example['sql']['agg']], values)
        reduced = functools.reduce(operators[example['agg']], values)

        ret = [indices[reduced[1]]]
        return ret, _Aggregation.NONE

    return indices, aggregation_op


def _get_float_answer(table,
                      answer_coordinates,
                      aggregation_op):
    """Applies operation to produce reference float answer."""
    if not answer_coordinates:
        if aggregation_op == _Aggregation.COUNT:
            return 0.0
        else:
            return _NAN

    # Count can support non-numeric answers.
    if aggregation_op == _Aggregation.COUNT:
        # return float(len(answer_coordinates))
        return int(len(answer_coordinates))

    # If we have just one answer, if float returns it or try a conversion.
    values = [table['rows'][i][j] for (i, j) in answer_coordinates]
    if len(answer_coordinates) == 1:
        try:
            return convert_to_float(values[0])
        except ValueError as e:
            if aggregation_op != _Aggregation.NONE:
                raise e

    if aggregation_op == _Aggregation.NONE:
        return None

    # Other aggregation only support numeric values. Bail out if we have strings.
    if not all((isinstance(v, (int, float)) for v in values)):
        return None

    if aggregation_op == _Aggregation.SUM:
        return float(sum(values))
    elif aggregation_op == _Aggregation.AVERAGE:
        return sum(values) / len(answer_coordinates)
    else:
        raise ValueError(f'Unknown aggregation: {aggregation_op}')


def _get_aggregation_name(aggregation):
    if aggregation == _Aggregation.NONE:
        return ''
    return aggregation.name


def _get_answer_text(table,
                     answer_coordinates,
                     float_answer):
    if float_answer is not None:
        return [str(float_answer)]
    return [str(table['rows'][r][c]) for r, c in answer_coordinates]



if __name__ == "__main__":  

	"""
		NOTE: Uncomment following lines to generate the vocabulary for mlm task
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default = "configs/mlm/gpt2.json", type = str, help = "Path to experiment configuration")
    # args = parser.parse_args()

    # with open(args.config, "r") as f:
    #     config = json.load(f)

    # config = process_config(config, args)


    # word_frequency_dict = construct_count_vocab(config)
    # filtered_words = sorted(list(dict(list(word_frequency_dict.items())[0: 150]).keys()))
    # filtered_words.remove("tabl")
    
    # with open(config.data.maskable_words_file, "wb") as f:
    #     pickle.dump(filtered_words, f)


	"""
		NOTE: Following lines of code for obtaining gold labels of wiki sql dataset
    """
    
	dataset = load_dataset("wikisql")
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	validation_dataset = dataset["validation"]
	
	train_answers = []
	for i in tqdm(range(len(train_dataset)), position = 0, leave = True, total = len(train_dataset)):
		table = train_dataset[i]["table"]
		sql = train_dataset[i]["sql"]
		answer_coordinates, aggregation_op = _get_answer_coordinates(table, sql)
		float_answer = _get_float_answer(table, answer_coordinates, aggregation_op)
		answer_text = _get_answer_text(table, answer_coordinates, float_answer)
		train_answers.append(answer_text)

	with open("datasets/wikisql_train_answers.pkl", "wb") as f:
		pickle.dump(train_answers, f)

	test_answers = []
	for i in tqdm(range(len(test_dataset)), position = 0, leave = True, total = len(test_dataset)):
		table = test_dataset[i]["table"]
		sql = test_dataset[i]["sql"]
		answer_coordinates, aggregation_op = _get_answer_coordinates(table, sql)
		float_answer = _get_float_answer(table, answer_coordinates, aggregation_op)
		answer_text = _get_answer_text(table, answer_coordinates, float_answer)
		test_answers.append(answer_text)

	with open("datasets/wikisql_test_answers.pkl", "wb") as f:
		pickle.dump(test_answers, f)

	validation_answers = []
	for i in tqdm(range(len(validation_dataset)), position = 0, leave = True, total = len(validation_dataset)):
		table = validation_dataset[i]["table"]
		sql = validation_dataset[i]["sql"]
		answer_coordinates, aggregation_op = _get_answer_coordinates(table, sql)
		float_answer = _get_float_answer(table, answer_coordinates, aggregation_op)
		answer_text = _get_answer_text(table, answer_coordinates, float_answer)
		validation_answers.append(answer_text)

	with open("datasets/wikisql_validation_answers.pkl", "wb") as f:
		pickle.dump(validation_answers, f)