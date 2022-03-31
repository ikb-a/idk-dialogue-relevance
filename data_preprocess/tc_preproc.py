from nltk.tokenize import word_tokenize
from utils.CorpusRepr import Corpus, AnnotatedDialogue, save_corpus
import json
import os
from typing import List
from data_preprocess.usr_preproc_split import fix

def process_file(dataset, dialogues: List[AnnotatedDialogue]) -> None:
    rating_mapping = {"Excellent": 5, "Good": 4, "Passable": 3, "Not Good": 2, "Poor": 1}

    for exchange in dataset.values():
        try:
            context = []
            response = None
            rating = None

            for turn in exchange['content']:
                if response is not None:
                    context.append(response)
                response = ' '.join(word_tokenize(turn['message'].strip().lower()))
                response = fix(response, True)  # Undo splitting of contractions
                rating = turn['turn_rating']

            if response is None:
                continue

            rating = rating_mapping[rating]

            dialogues.append(AnnotatedDialogue(context, response, [], True, rating, []))
        except KeyError:
            pass


def main():
    os.makedirs('data/tc', exist_ok=True)

    if not os.path.exists('data/tc/TopicalChat_val.json'):
        print("Processing TopicalChat Val")
        dialogues = []
        c2 = Corpus("TopicalChat_val")
        with open(os.path.join('data', 'valid_freq.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, dialogues)

        with open(os.path.join('data', 'valid_rare.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, dialogues)

        c2.dialogues = dialogues
        save_corpus(c2, 'data/tc/TopicalChat_val.json')

    if not os.path.exists('data/tc/TopicalChat_train.json'):
        print("Processing TopicalChat Train")
        dialogues = []
        c3 = Corpus("TopicalChat_train")
        with open(os.path.join('data', 'train.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, dialogues)

        c3.dialogues = dialogues
        save_corpus(c3, 'data/tc/TopicalChat_train.json')
