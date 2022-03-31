"""
Split HUMOD file into train, val, and test. Also merge ratings into single value.
"""
import numpy as np
import json
from nltk.tokenize import word_tokenize
import os

if __name__ == '__main__':
    HUMOD_PATH = "../../HUMOD_v2.0.json"

    with open(HUMOD_PATH, 'r') as humod_file:
        humod = json.load(humod_file)

    humod = humod["dialogues"]

    contexts= []
    responses = []
    scores = []

    for dial in humod:
        contexts.append(dial["context"])
        responses.append(dial["response"])
        scores.append(np.average(np.array(dial["resp_annot"])))

    contexts = [word_tokenize(' '.join(c).strip().lower()) for c in contexts]
    responses = [' '.join(word_tokenize(r.strip().lower())) for r in responses]

    if not os.path.exists('humod_train_ctx.txt'):
        with open("humod_train_ctx.txt", 'w') as out:
            for c in contexts[:3750]:
                print(" ".join(c), file=out)

        with open("humod_train_res.txt", 'w') as out:
            for c in responses[:3750]:
                print(c, file=out)

        with open("humod_valid_ctx.txt", 'w') as out:
            for c in contexts[3750:4250]:
                print(" ".join(c), file=out)

        with open("humod_valid_res.txt", 'w') as out:
            for c in responses[3750:4250]:
                print(c, file=out)

        with open("humod_valid_human.txt", 'w') as out:
            for c in scores[3750:4250]:
                print(c, file=out)
    else:
        print('Skip train + valid -- exists')

    if not os.path.exists('humod_test_ctx.txt'):
        with open("humod_test_ctx.txt", 'w') as out:
            for c in contexts[4250:]:
                print(" ".join(c), file=out)

        with open("humod_test_res.txt", 'w') as out:
            for c in responses[4250:]:
                print(c, file=out)

        with open("humod_test_human.txt", 'w') as out:
            for c in scores[4250:]:
                print(c, file=out)
    else:
        print('Skip test -- exists')
