"""
Split HUMOD file into train, val, and test. Also merge ratings into single value.
"""

import numpy as np
from utils.CorpusRepr import load_corpus, Corpus, save_corpus
import os

def main():
    HUMOD_PATH = "data/HUMOD_v2.0.json"
    humod_corpus = load_corpus(HUMOD_PATH, load_annot=True)

    os.makedirs('data/humod')

    for d in humod_corpus.dialogues:
        d.resp_annot = np.average(np.array(d.resp_annot))
        d.distr_annots = [np.average(np.array(d.distr_annots))]

    c = Corpus('train_' + humod_corpus.name, humod_corpus.distractor_names)
    c.dialogues = humod_corpus.dialogues[:3750]
    save_corpus(c, 'data/humod/train_HUMOD.json')

    c = Corpus('val_' + humod_corpus.name, humod_corpus.distractor_names)
    c.dialogues = humod_corpus.dialogues[3750:4250]
    save_corpus(c, 'data/humod/val_HUMOD.json')

    c = Corpus('test_' + humod_corpus.name, humod_corpus.distractor_names)
    c.dialogues = humod_corpus.dialogues[4250:]
    save_corpus(c, 'data/humod/test_HUMOD.json')

