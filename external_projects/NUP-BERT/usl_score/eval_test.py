import argparse
import json
from Scorer import Scorer
from scipy.stats import pearsonr, spearmanr
import os
from argparse import Namespace
from statistics import mean, stdev
from tabulate import tabulate

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='USL-H inference script')
    parser.add_argument('--weight-dir', type=str, required=True, help='Path to NUP weight')
    parser.add_argument('--context-file', type=str, required=True, help='Path to context file. Each line is a context.')
    parser.add_argument('--response-file', type=str, required=True, help='Path to response file. Each line is a response.')
    parser.add_argument('--human-file', type=str, required=True, help='Path to human scores for response-file. Each line is a response.')
    parser.add_argument('--output-score', type=str, default='output_scores.json', help='Path to the score output')
    args = parser.parse_args()
    """

    models = {
        'h': [0,1,2],
        'tc': [3,4,5],
        'tc3750': [6,7,8]
    }

    SIG_LEVEL = 0.01

    ROOT = 'lines/'
    datapaths = {
        'h': (ROOT+'humod/humod_test_ctx.txt',
              ROOT+'humod/humod_test_res.txt',
              ROOT+'humod/humod_test_human.txt'),
        'p_dd': (ROOT+'pang_dd/pang_dd_ctx.txt',
              ROOT+'pang_dd/pang_dd_res.txt',
              ROOT+'pang_dd/pang_dd_human.txt'),
        'tc': (ROOT+'usr-tc-rel/tc_a_test_ctx_3750.txt',
               ROOT+'usr-tc-rel/tc_a_test_res_3750.txt',
               ROOT+'usr-tc-rel/tc_a_test_human_3750.txt'),
        'fed_c': (ROOT+'fed/fed_cor_ctx.txt',
                  ROOT+'fed/fed_cor_res.txt',
                  ROOT+'fed/fed_cor_human.txt'),
        'fed_r': (ROOT+'fed/fed_rel_ctx.txt',
                  ROOT+'fed/fed_rel_res.txt',
                  ROOT+'fed/fed_rel_human.txt'),
    }

    final_results1 = [['', 'HUMOD S', 'HUMOD P',
                      'USR-TC S', 'USR-TC P', 'P-DD S', 'P-DD P',
                       'FED-COR S', 'FED-COR P', 'FED-REL S', 'FED-REL P']]


    for exp_final in ['h', 'tc3750', 'tc']:  # Iterate over exp train data
        print('----------------------------------')
        print(exp_final)
        print(final_results1)
        print(tabulate(final_results1, tablefmt='latex'))
        print(tabulate(final_results1))

        final_results1.append([exp_final])

        all_sig_s = {'h': True, 'tc': True, 'p_dd': True, 'fed_c': True, 'fed_r': True}
        some_sig_s = {'h': False,  'tc': False, 'p_dd': False, 'fed_c': False, 'fed_r': False}
        all_sig_p = {'h': True,  'tc': True, 'p_dd': True, 'fed_c': True, 'fed_r': True}
        some_sig_p = {'h': False,  'tc': False, 'p_dd': False, 'fed_c': False, 'fed_r': False}
        pearson = {}
        spearman = {}

        for rep, ver in enumerate(models[exp_final]):  # Iterate over trial
            print(f"EVAL {exp_final} ver{ver}")


            # Load checkpoint with better validation loss
            for test_dataset in ['fed_c', 'fed_r', 'tc', 'p_dd', 'h']:
                target_save_dir = f'exp_final/{exp_final}/{test_dataset}/{rep}/'

                """
                if os.path.exists(target_save_dir):
                    print('SKIPPING - exists')
                    continue
                else:
                    os.makedirs(target_save_dir, exist_ok=True)
                """
                os.makedirs(target_save_dir, exist_ok=True)

                try:
                    args=Namespace(weight_dir=f'exp_final/{exp_final}/lightning_logs/version_{ver}/checkpoints/epoch=0.ckpt',
                                   context_file=datapaths[test_dataset][0],
                                   response_file=datapaths[test_dataset][1],
                                   human_file=datapaths[test_dataset][2],
                                   output_score=target_save_dir + 'output.json')
                    scorer = Scorer(args)
                except FileNotFoundError:
                    args=Namespace(weight_dir=f'exp_final/{exp_final}/lightning_logs/version_{ver}/checkpoints/epoch=1.ckpt',
                                   context_file=datapaths[test_dataset][0],
                                   response_file=datapaths[test_dataset][1],
                                   human_file=datapaths[test_dataset][2],
                                   output_score=target_save_dir + 'output.json')
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
                #assert(len(human_scores) == 30 or len(human_scores) == 500 )

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

                    all_sig_p[test_dataset] = s2 < SIG_LEVEL and all_sig_p[test_dataset]
                    all_sig_s[test_dataset] = s1 < SIG_LEVEL and all_sig_s[test_dataset]
                    some_sig_p[test_dataset] = s2 < SIG_LEVEL or some_sig_p[test_dataset]
                    some_sig_s[test_dataset] = s1 < SIG_LEVEL or some_sig_s[test_dataset]

                if test_dataset in pearson:
                    pearson[test_dataset].append(c2)
                    spearman[test_dataset].append(c1)
                else:
                    pearson[test_dataset] = [c2]
                    spearman[test_dataset] = [c1]

                print (f'[!] evaluation complete. output to {args.output_score}')

        def format(scores, all_sig, some_sig):
            return f'{mean(scores):.2f}{"*" if all_sig else "+" if some_sig else ""} ({stdev(scores):.2f})'

        for test_dataset in ['h', 'tc', 'p_dd', 'fed_c', 'fed_r']:
            final_results1[-1] += [format(spearman[test_dataset], all_sig_s[test_dataset], some_sig_s[test_dataset]),
                                   format(pearson[test_dataset], all_sig_p[test_dataset], some_sig_p[test_dataset])]

    print('========================================')
    print(final_results1)
    print(tabulate(final_results1, tablefmt='latex'))
    print(tabulate(final_results1))
