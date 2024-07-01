# CABINET: CONTENT RELEVANCE BASED NOISE REDUCTION FOR TABLE QUESTION ANSWERING

Table understanding capability of Large Language Models (LLMs) has been extensively studied through the task of question-answering (QA) over tables. Typically, only a small part of the whole table is relevant to derive the answer for a given question. The irrelevant parts act as noise and are distracting information, resulting in sub-optimal performance due to the vulnerability of LLMs to noise. To mitigate this, we propose CABINET (Content RelevAnce-Based NoIse ReductioN for TablE QuesTion-Answering) – a framework to enable LLMs to focus on relevant tabular data by suppressing extraneous information. CABINET comprises an Unsupervised Relevance Scorer (URS), trained differentially with the QA LLM, that weighs the table content based on its relevance to the input question before feeding it to the question-answering LLM (QA LLM). To further aid the relevance scorer, CABINET employs a weakly supervised module that generates a parsing statement describing the criteria of rows and columns relevant to the question and highlights the content of corresponding table cells. CABINET significantly outperforms various tabular LLM baselines, as well as GPT3-based
in-context learning methods, is more robust to noise, maintains outperformance on tables of varying sizes, and establishes new SoTA performance on WikiTQ, FeTaQA, and WikiSQL datasets.

### File Description

This repository contains codes for some baselines and our proposed method. The details about the file and directory structure can be found below.

#### Baselines

- [OmniTab](./OmniTab): This directory contains the codebase for experiments and methods proposed in the paper titled [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://aclanthology.org/2022.naacl-main.68.pdf)
- [PASTA](./PASTA): This directory contains the codebase for experiments and methods proposed in the paper titled [PASTA: Table-Operations Aware Fact Verification via Sentence-Table Cloze Pre-training](https://aclanthology.org/2022.emnlp-main.331.pdf)
- [ReasTAP](./ReasTAP): This directory contains the codebase for experiments and methods proposed in the paper titled [REASTAP: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples](https://aclanthology.org/2022.emnlp-main.615.pdf)
- [TAPEX](./TAPEX): This directory contains the codebase for experiments and methods proposed in the paper titled [TAPEX: TABLE PRE-TRAINING VIA LEARNING A NEURAL SQL EXECUTOR](https://arxiv.org/pdf/2107.07653.pdf)

#### Our Code

- [configs](./configs): This directory contains the training and evaluation configuration for the different tasks and datasets explored. Specific details can be found [here]().

- [data](./data): This directory consists of python scripts to create PyTorch Dataset for different datasets and preprocessing utilities used for experiments

- [notebooks](./data): This directory contains python notebooks that were used carry out several ablations (performance of methods with respect to size, performance with perturbed tables etc.) and analyses over CABINET and the three benchmark datasets .

- [src](./src): This directory contains the code for different models used to conduct experiments

- [utils](./utils): This directory contains the code for creating a PyTorch Lightning trainer, some helper functions used for creating model instances, dataloader instances and evaluation scripts. It also has the scripts to generate the parsing statement, and further generate the highlighted cells for a question-table pair. We also provide the scripts to some random explorations as well.

- [main.py](./main.py): This file acts as a wrapper for calling several functions and classes for training and evaluating the model

### Setup the environment  

```
  conda env create -f environment.yml
  conda activate tabllm
```

### Experiments

Please download the datasets from [here](https://drive.google.com/drive/folders/1Jl0poE5EDflVccDJZ8JNEzmQOZ_D_z-t?usp=drive_link)
Please download the checkpoints from [here](https://drive.google.com/drive/folders/1h7BkGdJqxWYL1IHFj7N7efm36Lv9bZN2?usp=sharing)

To run the experiments and train the model with a certain config, run the following command
```
  python main.py --config <config_path>
```

To evaluate the trained model on a particular dataset, run the following command
```
  python evaluate.py --config <config_path> --device <device_name> -- --ckpt_path <checkpoint_path>
```

For CABINET, set the ```<config_path>``` as follows for the different datasets
- For WikiTQ, ```<config_path> = configs/wiki_tq_clustering_and_highlighting/tapex.json```
- For WikiSQL, ```<config_path> = configs/wiki_sql_clustering_and_highlighting/tapex.json```
- For FeTaQA, ```<config_path> = configs/feta_qa_clustering_and_highlighting/tapex.json```

### Citation

If you find this work useful and relevant to your research, please cite it. 
```
  @inproceedings{
    patnaik2024cabinet,
    title={{CABINET}: Content Relevance-based Noise Reduction for Table Question Answering},
    author={Sohan Patnaik and Heril Changwal and Milan Aggarwal and Sumit Bhatia and Yaman Kumar and Balaji Krishnamurthy},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=SQrHpTllXa}
  }
```
### Contact
For questions related to this code, please raise an issue and you can mail us at `sohanpatnaik106@gmail.com`.
