import torch
from transformers import BertForNextSentencePrediction, BertTokenizer
from torch.utils.data import Dataset
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
import argparse
from tqdm import tqdm
from utils.CorpusRepr import load_corpus
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn.functional import softmax

BATCHSIZE = 6
MARGIN = 0.4
MAX_LEN = 512

LR = 0.001

DEVICE = 'cuda:1'

ROOT = 'data/'
ROOT2 = 'data/'
# Map enum to train, val, test set paths
DATASET_PATHS = {
    'HUMOD': (ROOT2+'humod/train_HUMOD.json',
                        ROOT2+'humod/val_HUMOD.json',
                        ROOT2+'humod/test_HUMOD.json'),
    'USR_TC_REL': (ROOT + 'tc/TopicalChat_train.json',
                             ROOT2 + 'usr_tc/usr_tc_rel_val.json',
                             ROOT2 + 'usr_tc/usr_tc_rel_test.json'),
    'P_DD': (None,
             None,
             ROOT2 + 'pang_dd/pang_dd.json'),
    'FED_COR': (None,
                None,
                ROOT2 + 'fed/fed_cor.json'),
    'FED_REL': (None,
                None,
                ROOT2 + 'fed/fed_rel.json'),
}

"""Just use pretrained BERT NSP as a relevance measure"""
def main(dataset, load_test_set):
    if not load_test_set:
        save_dir = "exp/BERT_NSP"+dataset
    else:
        save_dir = "exp/BERT_NSP_test_"+dataset

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_dir + f'/corr.txt'):
        print("Results already exist, terminating")
        exit()

    if not load_test_set:
        assert dataset not in ['P_DD', 'FED_COR', 'FED_REL']
        _, val_corpus, _ = DATASET_PATHS[dataset]
    else:
        _, _, val_corpus = DATASET_PATHS[dataset]
    val_corpus = load_corpus(val_corpus, load_annot=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_distractors = len(val_corpus.distractor_names)




    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval().to(DEVICE)

    def perform_nsp(prompt, next_sentence):
        encoding = tokenizer.encode_plus(prompt, next_sentence, return_tensors='pt', max_length=512).to(DEVICE)
        with torch.no_grad():
            bert_out = model.bert(**encoding)
            pooled_output = bert_out[1]
            logits = model.cls(pooled_output)
            logits = logits[0]  # result is tensor with 2 values
            return softmax(logits, dim=0)[0].item()


    human_rel = []
    nsp_rel = []
    for j, x in tqdm(enumerate(val_corpus.dialogues)):
        prompt = " ".join(x.context)
        true_resp = x.response
        human_rel.append(x.resp_annot)
        nsp_rel.append(perform_nsp(prompt, true_resp))

        for i in range(num_distractors):
            distr_resp = x.distractors[i]
            human_rel.append(x.distr_annots[i])
            nsp_rel.append(perform_nsp(prompt, distr_resp))

        #if j%10 == 0:
        #    print(f"In progress Spearman Corr: {spearmanr(nsp_rel, human_rel)}")

    coef1, p1 = pearsonr(nsp_rel, human_rel)
    print("Pearson Rho: {} p={}".format(coef1, p1))
    print(f"Spearman Tau: {spearmanr(nsp_rel, human_rel)}")

    with open(save_dir + "/val_corr.txt", 'a') as outfile:
        print("Pearson Rho: {} p={}".format(coef1, p1), file=outfile)
        print(f"Spearman Tau: {spearmanr(nsp_rel, human_rel)}", file=outfile)

    with open(save_dir + "/predictions.txt", 'a') as outfile:
        for pred in nsp_rel:
            print(pred, file=outfile)

    with open(save_dir + "/results.csv", 'a') as outfile:
        print('"spearman","spear_sig","pearson","pear_sig"', file=outfile)
        coef2, p2 = spearmanr(nsp_rel, human_rel)
        print(f"{coef2},{p2},{coef1},{p1}", file=outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="What dataset to load; P_DD & FED only has test split", choices=['HUMOD', 'USR_TC_REL', 'P_DD', 'FED_COR', 'FED_REL'])
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print('-----------------------------------------------')
    print("RUNNING BERT COHERENCE MAX-POOL")
    print(args)
    main(args.dataset, args.test)
