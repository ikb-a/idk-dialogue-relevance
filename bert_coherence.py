import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
import argparse
from tqdm import tqdm
from utils.CorpusRepr import load_corpus
from torch.utils.data import TensorDataset
import torch.nn as nn

BATCHSIZE = 6
MARGIN = 0.4
MAX_LEN = 512

LR = 0.001

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


"""
Just compute cosine similarity of NUP transform of CLS
 embeddings under BERT
 """
def main(dataset: str, load_test_set):
    if not load_test_set:
        save_dir = "exp/BERT_coh"+dataset
    else:
        save_dir = "exp/BERT_coh_test_"+dataset

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_dir + f'/val_corr.txt'):
        print("Results already exist, terminating")
        exit()

    dev = "cuda" #cuda
    device = torch.device(dev)

    if not load_test_set:
        assert dataset != 'P_DD'
        _, val_corpus, _ = DATASET_PATHS[dataset]
    else:
        _, _, val_corpus = DATASET_PATHS[dataset]
    val_corpus = load_corpus(val_corpus, load_annot=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_distractors = len(val_corpus.distractor_names)

    # Get every response's' context
    val_X1 = [tokenizer.encode_plus(" ".join(x.context), max_length=MAX_LEN, pad_to_max_length=True,
                                   return_tensors='pt') for x in val_corpus.dialogues]
    for i in range(num_distractors):
        val_X1 += [tokenizer.encode_plus(" ".join(x.context), max_length=MAX_LEN,
                                        pad_to_max_length=True, return_tensors='pt') for x in val_corpus.dialogues]

    # Get every response
    val_X2 = [tokenizer.encode_plus(x.response, max_length=MAX_LEN, pad_to_max_length=True,
                                    return_tensors='pt') for x in val_corpus.dialogues]
    for i in range(num_distractors):
        val_X2 += [tokenizer.encode_plus(x.distractors[i], max_length=MAX_LEN,
                                         pad_to_max_length=True, return_tensors='pt') for x in val_corpus.dialogues]

    # Get human score for every response
    val_Y = [x.resp_annot for x in val_corpus.dialogues]
    for i in range(num_distractors):
        val_Y += [x.distr_annots[i] for x in val_corpus.dialogues]

    num_val_dialogues = len(val_X1)
    val_data_dict = {'tok1': torch.zeros((num_val_dialogues, MAX_LEN)),
                     'att1': torch.zeros((num_val_dialogues, MAX_LEN)),
                     'ty1': torch.zeros((num_val_dialogues, MAX_LEN)),
                     'tok2': torch.zeros((num_val_dialogues, MAX_LEN)),
                     'att2': torch.zeros((num_val_dialogues, MAX_LEN)),
                     'ty2': torch.zeros((num_val_dialogues, MAX_LEN))}
    for i, (v1, v2) in enumerate(zip(val_X1, val_X2)):
        val_data_dict['tok1'][i], val_data_dict['att1'][i], val_data_dict['ty1'][i] = \
            v1['input_ids'], v1['attention_mask'], v1['token_type_ids']
        val_data_dict['tok2'][i], val_data_dict['att2'][i], val_data_dict['ty2'][i] = \
            v2['input_ids'], v2['attention_mask'], v2['token_type_ids']

    val_dataset = TensorDataset(val_data_dict['tok1'], val_data_dict['att1'], val_data_dict['ty1'],
                                val_data_dict['tok2'], val_data_dict['att2'], val_data_dict['ty2'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=1)
    print("Loading Model")

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)
    torch.cuda.empty_cache()

    # Get validation loss
    print("Running")
    cos = nn.CosineSimilarity(dim=1)
    with torch.no_grad():
        predictions = []
        for i, data in enumerate(tqdm(val_loader)):
            data = [x.to(torch.int64).to(device) for x in data]
            tok1, att1, ty1, tok2, att2, ty2 = data
            _, feats1 = bert_model(input_ids=tok1, attention_mask=att1, token_type_ids = ty1)
            _, feats2 = bert_model(input_ids=tok2, attention_mask=att2, token_type_ids = ty2)


            val_pred = cos(feats1, feats2)
            val_pred = val_pred.detach().cpu().numpy()
            val_pred = np.squeeze(val_pred)
            val_pred = val_pred.tolist()
            predictions += val_pred
        coef1, p1 = pearsonr(predictions, val_Y)
        print("Pearson Rho: {} p={}".format(coef1, p1))
        print(f"Spearman Tau: {spearmanr(predictions, val_Y)}")

        with open(save_dir + "/val_corr.txt", 'a') as outfile:
            print("Pearson Rho: {} p={}".format(coef1, p1), file=outfile)
            print(f"Spearman Tau: {spearmanr(predictions, val_Y)}", file=outfile)

        with open(save_dir + "/predictions.txt", 'a') as outfile:
            for pred in predictions:
                print(pred, file=outfile)

        with open(save_dir + "/results.csv", 'a') as outfile:
            print('"spearman","spear_sig","pearson","pear_sig"', file=outfile)
            coef2, p2 = spearmanr(predictions, val_Y)
            print(f"{coef2},{p2},{coef1},{p1}", file=outfile)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="What dataset to load; P_DD & FED only has test split", choices=['HUMOD', 'USR_TC_REL', 'P_DD', 'FED_COR', 'FED_REL'])
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print("RUNNING BERT COHERENCE CLS FEAT")
    print(args)
    main(args.dataset, args.test)
