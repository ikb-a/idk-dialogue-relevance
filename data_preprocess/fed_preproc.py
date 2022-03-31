import pandas as pd
import numpy as np
import json
from utils.CorpusRepr import Corpus, AnnotatedDialogue, save_corpus
import os

def clear_prefix(string: str) -> str:
    assert ('System: ' in string and 'User: ' not in string) or \
           ('System: ' not in string and 'User: ' in string)
    string = string.replace('System: ', '')
    string = string.replace('User: ', '')
    return string

def make_and_save_corpus(rel_dialogues, save_name):
    corpus = Corpus("fed_rel", [])
    corpus.dialogues = rel_dialogues
    os.makedirs('data/fed/', exist_ok=True)
    save_corpus(corpus, os.path.join('data','fed', save_name))

def clean_scores(ratings):
    clean = [r for r in ratings if isinstance(r, int)]
    if len(clean) != len(ratings):
        print(ratings)
        return clean
    return ratings

def main():
    with open('data/fed_data.json', 'r') as infile:
        fed_dataset = json.load(infile)

    rel_dialogues=[]
    cor_dialogues=[]
    for turn in fed_dataset:
        if "response" not in turn:
            continue  # This is not a turn eval, but a conv eval -- skip
        context = turn["context"]
        context = context.split('\n')
        context = [clear_prefix(c) for c in context]
        response = clear_prefix(turn["response"])
        turn["annotations"]['Relevant'] = clean_scores(turn["annotations"]['Relevant'])
        rel_score = sum(turn["annotations"]['Relevant'])/len(turn["annotations"]['Relevant'])
        turn["annotations"]['Correct'] = clean_scores(turn["annotations"]['Correct'])
        cor_score = sum(turn["annotations"]['Correct'])/len(turn["annotations"]['Correct'])
        rel_dialogues.append(AnnotatedDialogue(context, response, [], True, rel_score, []))
        cor_dialogues.append(AnnotatedDialogue(context, response, [], True, cor_score, []))

    make_and_save_corpus(rel_dialogues, "fed_rel.json")
    make_and_save_corpus(cor_dialogues, "fed_cor.json")


if __name__ == '__main__':
    main()
