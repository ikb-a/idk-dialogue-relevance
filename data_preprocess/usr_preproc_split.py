"""
Convert the USR Topical Chat corpora into a Corpus object, and save as .json files.
Also, perform split into train & val sets.

Save relevance (well, "Maintains Context") rather than quality annotations
"""

from utils.CorpusRepr import Corpus, AnnotatedDialogue, save_corpus
import json
import os
from typing import List, Dict
import numpy as np


def fix(string: str, fix_contr: bool = True) -> str:
    if fix_contr:
        string = string.replace(" n't", "n't")
        string = string.replace("i 'm", "i'm")
        string = string.replace(" 're", "'re")
        string = string.replace(" 's", "'s")
        string = string.replace(" 've", "'ve")
        string = string.replace(" 'll", "'ll")
        string = string.replace(" 'd", "'d")
    return string

def convert_dict_to_corpus(dataset: List[Dict[str, object]], name: str, key: Dict[str, int],
                           distractor_labels: List[str], fix_contr = False) -> Corpus:
    """
    Convert a USR dataset into a Corpus object.

    Data is already in lower case and has been tokenized (punctuation split).

    The personachat data has split contractions, the topical chat data does not;
    if <fix_contr> then we'll undo the split to contractions

    <dataset> is the USR dataset, as read in via json.load
    <name> is the name to give to the corpus (and also will be used for filename)
    <key> maps names of model in the USR dataset file to the corresponding index of <distractor_labels>
    <distractor_labels> are the labels (and order) in which distractors should go into the Corpus
    """
    dialogues = []

    for exchange in dataset:
        context = exchange["context"].strip().split("\n")
        context = [fix(c, fix_contr) for c in context]
        gold_response = None
        distractors = [None] * len(distractor_labels)

        gold_response_annot = None
        distractors_annot = [None] * len(distractor_labels)

        for response in exchange["responses"]:
            if response["model"] == "Original Ground Truth":
                gold_response = fix(response["response"], fix_contr)
                gold_response_annot = np.average(np.array(response["Maintains Context"]))
            else:
                distractors[key[response["model"]]] = fix(response["response"], fix_contr)
                distractors_annot[key[response["model"]]] = np.average(np.array(response["Maintains Context"]))

        assert gold_response is not None and gold_response_annot is not None
        assert None not in distractors
        assert None not in distractors_annot

        dialogues.append(AnnotatedDialogue(context, gold_response, distractors, True,
                                           gold_response_annot, distractors_annot))

    # Models: "Original Ground Truth" "KV-MemNN" "Seq2Seq" "Language Model" "New Human Generated"

    corpus = Corpus(name, distractor_labels)
    corpus.dialogues = dialogues
    return corpus


def main():
    # topical chat
    key_tc = {"Argmax Decoding": 0, "Nucleus Decoding (p = 0.3)": 1,
              "Nucleus Decoding (p = 0.5)": 2, "Nucleus Decoding (p = 0.7)":3, "New Human Generated": 4}
    distr_labels_tc = ["Argmax Decoding", "NucleusDecoding_p0.3",
                       "NucleusDecoding_p0.5", "Nucleus Decoding_p0.7", "NewHumanGenerated"]

    to_process = [('tc_usr_data.json', 'usr_tc', key_tc, distr_labels_tc, True)]

    for filename, name, key, labels, fix_c in to_process:
        with open(os.path.join('data', filename), 'r') as infile:
            dataset = json.load(infile)

        # NOTE: USR TC data splits contractions, We undo the splitting of 
        # contractions on the TC data for consistency
        os.makedirs(f'data/{name}', exist_ok=True)
        save_corpus(convert_dict_to_corpus(dataset[:30], name+"_rel_val", key, labels, fix_c), f"data/{name}/{name}_rel_val.json")
        save_corpus(convert_dict_to_corpus(dataset[30:], name+"_rel_test", key, labels, fix_c), f"data/{name}/{name}_rel_test.json")
