import pandas as pd
import numpy as np
from utils.CorpusRepr import Corpus, AnnotatedDialogue, save_corpus
import os

def main():
    df = pd.read_csv('data/context_data_release.csv', header=None)
    cols_as_np = df[df.columns[3:]].to_numpy()
    print(cols_as_np.shape)
    scores = np.average(cols_as_np, axis=1)
    dialogues=[]
    for index, row in df.iterrows():
        # Convert to ASCII, and undo extra space
        context=row[1].replace(' ’ ', "'").replace(" ‘ ", "'")
        response = row[2].replace(' ’ ', "'").replace(" ‘ ", "'")
        dialogues.append(AnnotatedDialogue([context], response, [], True, scores[index], []))


    corpus = Corpus("pang_dd", [])
    corpus.dialogues = dialogues

    os.makedirs('data/pang_dd/', exist_ok=True)
    save_corpus(corpus, os.path.join('data','pang_dd', "pang_dd.json"))

if __name__ == '__main__':
    main()
