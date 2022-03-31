import json
import os
from tabulate import tabulate
from scipy.stats import pearsonr, spearmanr

def process(datasets_list):
    for metric in ['grade', 'dynaeval']:
        header = []
        row = []
        for  dir, filename, childname in datasets_list:
            #print(f"{metric} {dir} {childname}")

            # Get human scores
            with open(os.path.join("data", f"IDK_{dir}_{childname}_data", f"IDK_{dir}_{childname}.json"), 'r') as infile:
                annotations = json.load(infile)
            human = annotations['scores']

            with open(os.path.join("outputs", metric, f"IDK_{dir}_{childname}_data", f"results.json"), 'r') as infile:
                annotations = json.load(infile)
            machine = annotations[metric]

            corr_spear, sig = spearmanr(human, machine)
            header.append(f"{dir}{childname} S")
            if sig < 0.01:
                row.append(f"*{corr_spear:.2f}")
            else:
                row.append(f"{corr_spear:.2f}")

            corr_pear, sig = pearsonr(human, machine)
            header.append(f"{dir}{childname} P")
            if sig < 0.01:
                row.append(f"*{corr_pear:.2f}")
            else:
                row.append(f"{corr_pear:.2f}")

        print(metric)
        print(tabulate([header, row]))
        print()
        print(tabulate([header, row], tablefmt="latex"))



def main():
    process([("humod", "test_HUMOD.json", "test"),
             ("usr_tc", "usr_tc_rel_test.json", "test"),
             ("pang_dd", "pang_dd.json", "test"),
             ("fed", "fed_cor.json", "cor"),
             ("fed", "fed_rel.json", "rel")])
    print("\n\n")
    process([("humod", "val_HUMOD.json", "val"),
             ("usr_tc", "usr_tc_rel_val.json", "val")])

if __name__ == '__main__':
    main()
