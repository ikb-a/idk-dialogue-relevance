"""
NOTE: experiments aren't entirely optimal; have to try and undo punctuation
changes that came with USR-TC and P-DD; also can't recover capitalization
data that isn't present with USR-TC.

Do however add the <|endoftext|> turn markers used by FED.
"""

import external_projects.FED.fed_hack as fed
import json
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

VERBOSE=True

def fix_punct(string: str):
    string = string.replace(' ,', ',')
    string = string.replace(' .', '.')
    string = string.replace(' ?', '?')
    string = string.replace(' !', '!')
    string = string.replace('\n', '')
    return string


if __name__ == '__main__':
    # Load model
    model, tokenizer = fed.load_models("microsoft/DialoGPT-large")

    """
    data_to_run = {'humod_val': 'data/humod/val_HUMOD.json',
                   'usr-tc_val': 'data/usr_tc/usr_tc_rel_val.json',
                   'humod_test': 'data/humod/test_HUMOD.json',
                   'usr-tc_test': 'data/usr_tc/usr_tc_rel_test.json',
                   'p-dd': 'data/pang_dd/pang_dd.json'}
    data_to_run = {'fed_rel': 'data/fed/fed_rel.json',
                   'fed_cor': 'data/fed/fed_cor.json'}
    """
    data_to_run = {'p-dd': 'data/pang_dd/pang_dd.json'}

    for dataset_name in data_to_run:
        with open(data_to_run[dataset_name], 'r') as infile:
            my_dataset = json.load(infile)

        human = []
        fed_relevant = []
        fed_correct = []
        fed_avg = []
        fed_avg_turn = []
        TURN = ["interesting", "engaging", "specific", "relevant", "correct", "semantically appropriate", "understandable", "fluent"]

        for dialogue_dat in tqdm(my_dataset['dialogues']):
            # TODO: Annoying dataset-specific processing
            # HUMOD is fine (no weird punctuation or capitalization removal)
            # Pang-DD and USR-TC have punctuation borks
            #  -- mostly undone by fix_puncht; BUT USR-TC also has extra
            #     spaces around double quotes that is retained
            # USR-TC is also all lowercase, but that cannot be fixed (data
            # came that way)
            if 'humod' not in dataset_name:
                preproc = lambda x: x
            else:
                preproc = fix_punct

            dialogue_dat['context'] = [preproc(x) for x in dialogue_dat['context']]
            dialogue_dat['response'] = preproc(dialogue_dat['response'])
            dialogue_dat['distractors'] = [preproc(x) for x in dialogue_dat['distractors']]



            # Evaluate
            #conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
            conv_context = "<|endoftext|> " + "<|endoftext|>".join(dialogue_dat['context'])

            for resp, human_score in zip([dialogue_dat['response']] + dialogue_dat['distractors'],
                                         [dialogue_dat['resp_annot']] + dialogue_dat['distr_annots']):
                conversation = conv_context + " <|endoftext|> " + resp
                human.append(human_score)
                scores = fed.evaluate(conversation,
                                      model,
                                      tokenizer)

                if VERBOSE:
                    print(conversation)
                    print(scores)

                fed_relevant.append(scores['relevant'])
                fed_correct.append(scores['correct'])
                if 'interesting' not in scores:
                    print(scores)
                fed_avg.append(sum([scores[metric_name] for metric_name in scores]))
                tmp_list = [scores[metric_name] for metric_name in TURN]
                fed_avg_turn.append(sum(tmp_list)/len(tmp_list))

                if VERBOSE:
                    print(fed_avg_turn[-1])

        print(dataset_name)
        print("Coef, p-value")
        print(f"Correct, Pearson: {pearsonr(human, fed_correct)}")
        print(f"Correct, Spearman: {spearmanr(human, fed_correct)}")
        print(f"Relevant, Pearson: {pearsonr(human, fed_relevant)}")
        print(f"Relevant, Spearman: {spearmanr(human, fed_relevant)}")
        print(f"FA Relevant, Pearson: {pearsonr(human, fed_avg)}")
        print(f"FA Relevant, Spearman: {spearmanr(human, fed_avg)}")
        print(f"FAT Relevant, Pearson: {pearsonr(human, fed_avg_turn)}")
        print(f"FAT Relevant, Spearman: {spearmanr(human, fed_avg_turn)}")

        with open('fed_per_avg_hack.txt', 'a') as outfile:
            print(dataset_name, file=outfile)
            print("Coef, p-value")
            print(f"Correct, Pearson: {pearsonr(human, fed_correct)}", file=outfile)
            print(f"Correct, Spearman: {spearmanr(human, fed_correct)}", file=outfile)
            print(f"Relevant, Pearson: {pearsonr(human, fed_relevant)}", file=outfile)
            print(f"Relevant, Spearman: {spearmanr(human, fed_relevant)}", file=outfile)        
            print(f"FA Relevant, Pearson: {pearsonr(human, fed_avg)}", file=outfile)
            print(f"FA Relevant, Spearman: {spearmanr(human, fed_avg)}", file=outfile)
            print(f"FAT Relevant, Pearson: {pearsonr(human, fed_avg_turn)}", file=outfile)
            print(f"FAT Relevant, Spearman: {spearmanr(human, fed_avg_turn)}", file=outfile)

