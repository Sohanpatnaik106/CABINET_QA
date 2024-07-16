
import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_metric

from utils import (
        set_seed, 
        process_config, 
        prepare_dataloaders, 
        prepare_models,
        to_value_list, 
        check_denotation
    )


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "./configs/wiki_tq_clustering_and_highlighting/tapex.json", type = str, help = "Path to experiment configuration")
    parser.add_argument("--device", default = "cuda:1", type = str, help = "Device to put model and inputs")
    parser.add_argument("--ckpt_path", default = "./checkpoints/cabinet_wikitq_ckpt.pt", type = str, help = "Path to model checkpoint")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config = process_config(config, args)
    set_seed(config.seed)

    if config.data.config_name is not None:
        dataset = load_dataset(config.data.data_path, config.data.config_name)
    else:
        dataset = load_dataset(config.data.data_path)

    if config.training.training_type == "descriptive_table_question_answering":
        sacrebleu = load_metric("sacrebleu")

    train_dataloader, validation_dataloader, test_dataloader, tokenizer = prepare_dataloaders(dataset, config)
    model = prepare_models(config)
    model.load_state_dict(torch.load(args.ckpt_path, map_location = "cpu"))

    model.to(args.device)

    count = 0
    total = 0
    for idx, batch in tqdm(enumerate(test_dataloader), position = 0, leave = True, total = len(test_dataloader)):
        input_ids, attention_mask, token_type_ids, decoder_input_ids, highlighted_cells, labels = batch

        actual_output_ids = decoder_input_ids.clone()
        output_ids = model.model.generate(input_ids = input_ids.to(args.device), max_new_tokens = config.tokenizer.output_max_length, 
                                            num_beams = 3, early_stopping = True, attention_mask = attention_mask.to(args.device), 
                                            highlighted_cells = highlighted_cells.to(args.device))


        predicted_sequence = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
        actual_sequence = tokenizer.batch_decode(actual_output_ids, skip_special_tokens = True)

        if config.training.training_type == "descriptive_table_question_answering":
            # NOTE: Compute SacreBLEU score - For FeTaQA dataset
            # actual_sequence = [[a] for a in actual_sequence]
            # predicted_sequence = [predicted_sequence]

            for a, p in zip(actual_sequence, predicted_sequence):
                
                res = sacrebleu.compute(predictions = [p], references = [[a]])
                count += res["score"]
                total += 1

        else:

            for a, p in zip(actual_sequence, predicted_sequence):
                a = [x.strip() for x in a.split(",")]
                p = [x.strip() for x in p.split(",")]

                pred = to_value_list(p)
                gold = to_value_list(a)

                correct = check_denotation(gold, pred)
                if correct:
                    count += 1
                
                total += 1

    if config.training.training_type == "descriptive_table_question_answering":
        print(f"Sacre BLEU: {count / total: .4f}")
    else:
        print(f"Accuracy: {count / total: .4f}")
