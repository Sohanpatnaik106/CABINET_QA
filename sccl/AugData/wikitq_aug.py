import os
import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from nlpaug.util import Action


from datasets import load_dataset

from tqdm import tqdm

import pandas as pd


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# def contextual_augment(data_source, data_target, textcol="text", aug_p=0.2, device1="cuda", device2="cuda"):
def contextual_augment(dataset, textcol="question", aug_p=0.2, device1="cuda", device2="cuda"):
    ### contextual augmentation 
    print(f"\n-----transformer_augment-----\n")
    augmenter1 = naw.ContextualWordEmbsAug(
        model_path='roberta-base', action="substitute", aug_min=1, aug_p=aug_p, device=device1)
    
    augmenter2 = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="substitute", aug_min=1, aug_p=aug_p, device=device2)
        
    # train_data = pd.read_csv(data_source)
    # train_text = train_data[textcol].fillna('.').astype(str).values
    # print("train_text:", len(train_text), type(train_text[0]))

    train_data = dataset[textcol]

    auglist1, auglist2 = [], []
    for txt in tqdm(train_data, position = 0, leave = True, total = len(train_data)):
        atxt1 = augmenter1.augment(txt)
        atxt2 = augmenter2.augment(txt)

        auglist1.append(str(atxt1[0]))
        auglist2.append(str(atxt2[0]))

    return auglist1, auglist2
        
    # train_data[textcol+"1"] = pd.Series(auglist1)
    # train_data[textcol+"2"] = pd.Series(auglist2)
    # train_data.to_csv(data_target, index=False)


    
    # for o, a1, a2 in zip(train_text[:5], auglist1[:5], auglist2[:5]):
    #     print("-----Original Text: \n", o)
    #     print("-----Augmented Text1: \n", a1)
    #     print("-----Augmented Text2: \n", a2)


def word_deletion(data_source, data_target, textcol="text", aug_p=0.2):
    ### wordnet based data augmentation
    print(f"\n-----word_deletion-----\n")
    aug = naw.RandomWordAug(aug_min=1, aug_p=aug_p)
    
    train_data = pd.read_csv(data_source)
    train_text = train_data[textcol].fillna('.').astype(str).values
    print("train_text:", len(train_text), type(train_text[0]))

    augtxts1, augtxts2 = [], []
    for txt in train_text:
        atxt = aug.augment(txt, n=2, num_thread=1)
        augtxts1.append(str(atxt[0]))
        augtxts2.append(str(atxt[1]))
        
    train_data[textcol+"1"] = pd.Series(augtxts1)
    train_data[textcol+"2"] = pd.Series(augtxts2)
    train_data.to_csv(data_target, index=False)
    
    for o, a1, a2 in zip(train_text[:5], augtxts1[:5], augtxts2[:5]):
        print("-----Original Text: \n", o)
        print("-----Augmentation1: \n", a1)
        print("-----Augmentation2: \n", a2)
        
        
def randomchar_augment(data_source, data_target, textcol="text", aug_p=0.2, augstage="post"):
    ### wordnet based data augmentation
    print(f"\n*****random char aug: rate--{aug_p}, stage: {augstage}*****\n")
    aug = nac.RandomCharAug(action="swap", aug_char_p=aug_p, aug_word_p=aug_p)
    
    train_data = pd.read_csv(data_source)
    if augstage == "init":
        train_text = train_data[textcol].fillna('.').astype(str).values
        print("train_text:", len(train_text), type(train_text[0]))

        augtxts1, augtxts2 = [], []
        for txt in train_text:
            atxt = aug.augment(txt, n=2, num_thread=1)
            augtxts1.append(str(atxt[0]))
            augtxts2.append(str(atxt[1]))

        train_data[textcol+"1"] = pd.Series(augtxts1)
        train_data[textcol+"2"] = pd.Series(augtxts2)
        train_data.to_csv(data_target, index=False)

        for o, a1, a2 in zip(train_text[:5], augtxts1[:5], augtxts2[:5]):
            print("-----Original Text: \n", o)
            print("-----Augmentation1: \n", a1)
            print("-----Augmentation2: \n", a2)
    else:
        train_text1 = train_data[textcol+"1"].fillna('.').astype(str).values
        train_text2 = train_data[textcol+"2"].fillna('.').astype(str).values
        
        augtxts1, augtxts2 = [], []
        for txt1, txt2 in zip(train_text1, train_text2):
            atxt1 = aug.augment(txt1, n=1, num_thread=1)
            atxt2 = aug.augment(txt2, n=1, num_thread=1)
            augtxts1.append(str(atxt1))
            augtxts2.append(str(atxt2))

        train_data[textcol+"1"] = pd.Series(augtxts1)
        train_data[textcol+"2"] = pd.Series(augtxts2)
        train_data.to_csv(data_target, index=False)

        for o1, a1, o2, a2 in zip(train_text1[:2], augtxts1[:2], train_text2[:2], augtxts2[:2]):
            print("-----Original Text1: \n", o1)
            print("-----Augmentation1: \n", a1)
            print("-----Original Text2: \n", o2)
            print("-----Augmentation2: \n", a2)
        
        

def augment_text(datadir="./", targetdir="./", dataset="wiki1m_unique", aug_p=0.1, augtype="trans_subst"):
    set_global_random_seed(0)
    device1=torch.cuda.set_device(0)
    device2=torch.cuda.set_device(1)
    
    # DataSource = os.path.join(datadir, dataset + ".csv")
    # DataTarget = os.path.join(targetdir, '{}_{}_{}.csv'.format(dataset, augtype, int(aug_p*100)))

    if augtype == "word_deletion":
        raise NotImplementedError
        augseq = word_deletion(DataSource, DataTarget, textcol="text", aug_p=aug_p)
    elif augtype == "trans_subst":
        auglist1, auglist2 = contextual_augment(dataset=dataset, textcol="question", aug_p=aug_p, device1=device1, device2=device2)
    elif augtype == "charswap":
        raise NotImplementedError
        augseq = randomchar_augment(DataSource, DataTarget, textcol="text", aug_p=aug_p, augstage="post")
    else:
        print("Please specify AugType!!")

    return auglist1, auglist2
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitablequestions')
    parser.add_argument('--augtype', type=str, default='ctxt_insertbertroberta')
    parser.add_argument('--aug_min', type=int, default=1) 
    parser.add_argument('--aug_p', type=float, default=0.2)
    parser.add_argument('--aug_max', type=int, default=10)
    parser.add_argument('--gpuid', type=int, default=0)
    args = parser.parse_args()


    datadir = "path-to-the-original-datasets"
    targetdir = "path-to-store-the-augmented-datasets"
    
    dataset = load_dataset(args.dataset)

    auglist1, auglist2 = augment_text(datadir=datadir, targetdir=targetdir, dataset=dataset["train"], aug_p=0.2, augtype="trans_subst")

    data_dict = {"question": list(dataset["train"]["question"]),
                 "augmentation1": auglist1,
                 "augmentation2": auglist2,
                 "label": [0] * len(auglist1)}

    df = pd.DataFrame(data_dict)
    df.to_csv("/home/sohanp/sccl/datasets/wikitq_train.csv", index = False)

    print("Saved the file")

    # for i, (question, aug1, aug2) in tqdm(enumerate(zip(dataset["train"]["question"], auglist1, auglist2)), position = 0, leave = True, total = len(auglist1)):


    # print(dataset["train"]["question"][10])
    # print(auglist1[10])
    # print(auglist2[10])
