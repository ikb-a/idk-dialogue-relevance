import json
import os
from nltk.tokenize import word_tokenize
import numpy as np

def process_file(dataset, ctxt_path, res_path, limit=None) -> None:
    saved_lines = 0

    for exchange in dataset.values():
        try:
            context = []
            response = None

            for turn in exchange['content']:
                if response is not None:
                    context.append(response)
                response = ' '.join(word_tokenize(turn['message'].strip().lower()))

            if response is None:
                continue

            with open(ctxt_path, 'a') as out_c:
                print(" ".join(context), file=out_c)
            with open(res_path, 'a') as out_r:
                print(response, file=out_r)

            saved_lines += 1
            if limit is not None and saved_lines >= limit:
                return
        except KeyError:
            pass


if __name__ == '__main__':
    # Note text already appers to be in lower case,
    # and split into tokens (but contractions not split)

    lines_printed = 0

    if not os.path.exists('tc_train_ctx_3750.txt') and not os.path.exists('tc_train_res_3750.txt'):
        with open(os.path.join('..', '..', 'train.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, 'tc_train_ctx_3750.txt', 'tc_train_res_3750.txt', 3750)
    else:
        print("Skipping train 3750 -- files exist")


    if not os.path.exists('tc_train_ctx.txt') and not os.path.exists('tc_train_res.txt'):
        with open(os.path.join('..', '..', 'train.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, 'tc_train_ctx.txt', 'tc_train_res.txt')
    else:
        print("Skipping train -- files exist")


    # Full TC valid
    if not os.path.exists('tc_valid_ctx.txt') and not os.path.exists('tc_valid_res.txt'):
        with open(os.path.join('..', '..', 'valid_freq.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, 'tc_valid_ctx.txt', 'tc_valid_res.txt')
        with open(os.path.join('..', '..',  'valid_rare.json'), 'r') as infile:
            tmp = json.load(infile)
        process_file(tmp, 'tc_valid_ctx.txt', 'tc_valid_res.txt')
    else:
        print("Skipping valid -- files exist")

    # USR-TC valid -- human only
    if not os.path.exists('tc_valid_ctx_3750.txt') and not os.path.exists('tc_valid_res_3750.txt'):
        with open('../../tc_usr_data.json', 'r') as infile:
            tmp = json.load(infile)[:30]  # Validation set is first 30

        dialogues = []

        for i, exchange in enumerate(tmp):
            # Ignore the random distractors
            context = exchange["context"].strip().split("\n")

            for response in exchange["responses"]:
                if response["model"] == "Original Ground Truth":
                    gold_response = response["response"].strip()

            with open('tc_valid_ctx_3750.txt', 'a') as out_ctx:
                print(" ".join(context), file=out_ctx)
            with open('tc_valid_res_3750.txt', 'a') as out_res:
                print(gold_response, file=out_res)
    else:
        print("Skipping valid 3750 -- files exist")

    # test split 
    if not os.path.exists('tc_a_test_ctx_3750.txt') and not os.path.exists('tc_a_test_res_3750.txt'):
        with open('../../tc_usr_data.json', 'r') as infile:
            tmp = json.load(infile)[30:]  # test set is last 30

        dialogues = []

        for i, exchange in enumerate(tmp):
            context = exchange["context"].strip().split("\n")

            for response in exchange["responses"]:
                with open('tc_a_test_ctx_3750.txt', 'a') as out_ctx:
                    print(" ".join(context), file=out_ctx)
                with open('tc_a_test_res_3750.txt', 'a') as out_res:
                    print(response["response"].strip(), file=out_res)
                response_annot = np.average(np.array(response["Maintains Context"]))
                with open('tc_a_test_human_3750.txt', 'a') as out_res:
                    print(response_annot, file=out_res)
    else:
        print("Skipping test human 3750 -- files exist")


    # valid split 
    if not os.path.exists('tc_a_valid_ctx_3750.txt') and not os.path.exists('tc_a_valid_res_3750.txt'):
        with open('../../tc_usr_data.json', 'r') as infile:
            tmp = json.load(infile)[:30]  # valid set is first 30

        dialogues = []

        for i, exchange in enumerate(tmp):
            context = exchange["context"].strip().split("\n")

            for response in exchange["responses"]:
                with open('tc_a_valid_ctx_3750.txt', 'a') as out_ctx:
                    print(" ".join(context), file=out_ctx)
                with open('tc_a_valid_res_3750.txt', 'a') as out_res:
                    print(response["response"].strip(), file=out_res)
                response_annot = np.average(np.array(response["Maintains Context"]))
                with open('tc_a_valid_human_3750.txt', 'a') as out_res:
                    print(response_annot, file=out_res)
    else:
        print("Skipping valid human 3750 -- files exist")
