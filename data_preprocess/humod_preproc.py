from utils.CorpusRepr import Corpus, AnnotatedDialogue, save_corpus

NAME = "HUMOD_v2.0"
DISTRACTORS = ["RANDOM"]
ORI_LOAD_PATH = "data/HUMOD_Dataset.TXT"
LOAD_PATH = "data/HUMOD_Dataset2.TXT"
import os


def get_resp_and_context(vals):
    response = vals[-8]
    response = response.replace('Speaker 1:', '')
    response = response.replace('Speaker 2:', '')
    response = response.strip()
    context = vals[1:-8]
    context = " ".join(context)
    context = context.replace(" 2:", "")
    context = context.replace(" 1:", "")
    context = context.split("Speaker")
    context = [x.strip() for x in context]
    context = context[1:]
    return response, context

def main():
    # HUMOD 2 corrects some errors in dataset, extra ':' in one of the
    # distractors for 8294/8295, and missing tab
    # another missing tab for 8297
    with open(ORI_LOAD_PATH, 'r') as infile:
        with open(LOAD_PATH, 'w') as outfile:
            for i, line in enumerate(infile):
                if i == 10830 - 1:
                    line = line.replace('yeah it was sometimes   1268', 'yeah it was sometimes\t1268')
                elif i == 10831 - 1:
                    line = line.replace("For me that's real  923", "For me that's real\t923")
                elif i == 24889 - 1:
                    line = line.replace("in how long its got to hold for.:", "in how long its got to hold for.")
                elif i == 24895 - 1:
                    line = line.replace("that is not my business 1179", "that is not my business\t1179")
                print(line, file=outfile, end='')


    dialogues = []
    last_odd_context = ""
    curr_dial = -1

    with open(LOAD_PATH, 'r') as f:
        f.readline() # Skip header
        line = f.readline()
        while line:
            vals = line.split('\t')
            dial_num = int(vals[0])
            assert(0 == int(vals[-7]))
            gold_resp, context = get_resp_and_context(vals)

            gold_annot = [int(vals[-3])]

            for i in range(2):
                line = f.readline()
                #print(line)
                vals = line.split('\t')
                assert(0 == int(vals[-7]))
                assert(dial_num == int(vals[0]))
                #print((gold_resp,context))
                #print(get_resp_and_context(vals))
                assert((gold_resp, context) == get_resp_and_context(vals))
                gold_annot.append(int(vals[-3]))

            distr_annot = []
            distr_response = None

            for i in range(3):
                line = f.readline()
                vals = line.split('\t')
                assert (1 == int(vals[-7]))
                assert (dial_num + 1 == int(vals[0]))
                distr_response, context_distr = get_resp_and_context(vals)
                #print((gold_resp, context_distr))
                assert(context == context_distr)
                distr_annot.append(int(vals[-3]))

            dialogues.append(AnnotatedDialogue(context, gold_resp, [distr_response], True, gold_annot, [distr_annot]))

            line = f.readline()

    corpus = Corpus(NAME, DISTRACTORS)
    corpus.dialogues = dialogues

    save_corpus(corpus, os.path.join('data', NAME + ".json"))
