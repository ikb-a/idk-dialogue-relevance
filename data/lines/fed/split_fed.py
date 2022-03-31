import json
from nltk.tokenize import word_tokenize


if __name__ == '__main__':
    with open('../../fed/fed_cor.json', 'r') as infile:
        data = json.load(infile)

    with open('fed_cor_ctx.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(' '.join(word_tokenize(' '.join(dial["context"]))).lower(), file=outfile)

    with open('fed_cor_res.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(' '.join(word_tokenize(dial["response"])).lower(), file=outfile)

    with open('fed_cor_human.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(dial["resp_annot"], file=outfile)

    # Repeat for FED REL
    with open('../../fed/fed_rel.json', 'r') as infile:
        data = json.load(infile)

    with open('fed_rel_ctx.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(' '.join(word_tokenize(' '.join(dial["context"]))).lower(), file=outfile)

    with open('fed_rel_res.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(' '.join(word_tokenize(dial["response"])).lower(), file=outfile)

    with open('fed_rel_human.txt', 'w') as outfile:
        for dial in data['dialogues']:
            print(dial["resp_annot"], file=outfile)
