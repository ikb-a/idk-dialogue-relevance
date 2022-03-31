import os
from tabulate import tabulate
from statistics import mean, stdev

TARGET_EPOCH = 2
file_num = TARGET_EPOCH if TARGET_EPOCH != 2 else ''

def main(results,test_datasets):
    files = os.listdir('exp')
    files.sort()

    def is_target(string):
        return string.startswith("HUMOD_") or string.startswith("USR_")

    files = [f for f in files if is_target(f)]



    for f in files:
        new_line = [f]
        for test_dataset in test_datasets:
            try:
                with open(os.path.join('exp',f,f'test_results{file_num}_{test_dataset}.csv'),'r') as infile:
                    results_csv = infile.readlines()
            except FileNotFoundError:
                results_csv = [None]

            results_csv = results_csv[1:]
            results_csv = [r for r in results_csv if len(r.split(',')) == 4]
            hp = []
            hs = []

            # Note: for ease of implementation, triplet loss is backwards
            # -- i.e., it predicts 0 for good matches and 1 for bad matches.
            # For consistency, BCE was implemented the same way.
            # Therefore, correlation must be negated for these.
            correction = 1
            if f.startswith("HUMOD_") or f.startswith("USR_"):
                correction = -1

            sig_p = False
            all_sig_p = True
            sig_s = False
            all_sig_s = True
            for line in results_csv:
                line = line.split(',')
                hs.append(correction*float(line[0]))
                hp.append(correction*float(line[2]))
                SIGNIF = 0.01
                sig_p = sig_p or (float(line[3]) < SIGNIF)
                sig_s = sig_s or (float(line[1]) < SIGNIF)
                all_sig_p = all_sig_p and (float(line[3]) < SIGNIF)
                all_sig_s = all_sig_s and (float(line[1]) < SIGNIF)

            symb_s = '*' if all_sig_s else '+' if sig_s else ''
            symb_p = '*' if all_sig_p else '+' if sig_p else ''

            #print(os.path.join('exp',f,f'test_results{file_num}_{test_dataset}.csv'))
            #print(hs)

            for sym, corrs in [(symb_s,hs), (symb_p, hp)]:
                if len(corrs) == 0:
                    new_line.append(f"-")
                elif len(corrs) == 1:
                    new_line.append(f"{corrs[0]:.2f}{sym}")
                else:
                    new_line.append(f"{mean(corrs):.2f}{sym} ({stdev(corrs):.2f})")

        results.append(new_line)
    print(f"TARGET EPOCH {TARGET_EPOCH}")
    print(tabulate(results))



    # Again for LaTeX
    print(tabulate(results, tablefmt='latex'))

if __name__ == '__main__':
    results = [["Name","HUMOD Spear", "HUMOD Pear", "TC Spear", "TC Pear", "P_DD Spear", "P_DD Pear"]]
    main(results, test_datasets=["HUMOD", "USR_TC_REL", "P_DD"])
    main([["Name","FED-Cor Spear", "FED-Cor Pear", "FED-Rel Spear", "FED-Rel Pear"]], test_datasets=["FED_COR", "FED_REL"])
