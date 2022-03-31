import argparse
import json
from Scorer import Scorer
from scipy.stats import pearsonr, spearmanr
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='USL-H inference script')
    parser.add_argument('--weight-dir', type=str, required=True, help='Path to NUP weight')
    parser.add_argument('--context-file', type=str, required=True, help='Path to context file. Each line is a context.')
    parser.add_argument('--response-file', type=str, required=True, help='Path to response file. Each line is a response.')
    parser.add_argument('--human-file', type=str, required=True, help='Path to human scores for response-file. Each line is a response.')
    parser.add_argument('--output-score', type=str, default='output_scores.json', help='Path to the score output')

    args = parser.parse_args()
    scorer = Scorer(args)

    contexts = []
    responses = []
    with open(args.context_file) as f:
       for line in f:
           contexts.append(line)
       f.close()
    with open(args.response_file) as f:
       for line in f:
           responses.append(line)
       f.close()

    avg_score, scores = scorer.get_scores(contexts, responses, normalize=True)
    print (avg_score)

    ordered_nup_scores = []

    with open(args.output_score, 'w') as f:
        for score in scores:
            ordered_nup_scores.append(score["nup"])
            json_text = json.dumps(score)
            f.write(json_text + '\n')
        f.close()

    with open(args.human_file, 'r') as human:
        human_scores = human.readlines()
    human_scores = [h.strip() for h in human_scores if h.strip() != ""]
    human_scores = [float(h) for h in human_scores]

    print(len(human_scores))
    print(len(ordered_nup_scores))

    with open(os.path.join(os.path.dirname(args.output_score),"corr_" + os.path.basename(args.human_file)), 'w') as outfile:
        print(pearsonr(ordered_nup_scores, human_scores), file=outfile)
        print(spearmanr(ordered_nup_scores, human_scores), file=outfile)
        print(pearsonr(ordered_nup_scores, human_scores))
        print(spearmanr(ordered_nup_scores, human_scores))

    with open(os.path.join(os.path.dirname(args.output_score),"corr_nicer_" + os.path.basename(args.human_file)) + '.csv', 'w') as outfile:
        c1, s1 = spearmanr(ordered_nup_scores, human_scores)
        c2, s2 = pearsonr(ordered_nup_scores, human_scores)
        print('spearman,spearman_sig,pearson,pearson_sig', file=outfile)
        print(f"{c1},{s1},{c2},{s2}", file=outfile)



    print (f'[!] evaluation complete. output to {args.output_score}')
