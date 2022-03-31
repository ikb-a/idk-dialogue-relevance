import json
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
    with open('../../pang_dd/pang_dd.json', 'r') as infile:
        data = json.load(infile)

    with open('pang_dd_ctx.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(' '.join(word_tokenize(' '.join(dial["context"]))).lower(), file=outfile)

    with open('pang_dd_res.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(' '.join(word_tokenize(dial["response"])).lower(), file=outfile)

    with open('pang_dd_human.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(dial["resp_annot"], file=outfile)
