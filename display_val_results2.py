import os
from tabulate import tabulate
from statistics import mean, stdev

TARGET_EPOCH = 2
file_num = TARGET_EPOCH if TARGET_EPOCH != 2 else ''

def main():
    files = os.listdir('exp')
    files.sort()

    def is_target(string):
        return string.startswith("HUMOD_") or string.startswith("USR_")

    files = [f for f in files if is_target(f)]

    results = [["Name","HUMOD Spear", "HUMOD Pear", "TC Spear", "TC Pear"]]

    for f in files:
        try:
            with open(os.path.join('exp',f,f'results{file_num}.csv'),'r') as infile:
                results_csv = infile.readlines()
        except FileNotFoundError:
            results_csv = [None]

        results_csv = results_csv[1:]
        results_csv = [r for r in results_csv if len(r.split(',')) == 8]
        hp = []
        hs = []
        tp = []
        ts = []

        # Note: for ease of implementation, triplet loss is backwards
        # -- i.e., it predicts 0 for good matches and 1 for bad matches.
        # For consistency, BCE was implemented the same way.
        # Therefore, correlation must be negated for these.
        correction = 1
        if f.startswith("HUMOD_") or f.startswith("USR_"):
            correction = -1

        some_sig = {'hs': False, 'hp': False, 'ts': False, 'tp': False}
        all_sig = {'hs': True, 'hp': True, 'ts': True, 'tp': True}

        for line in results_csv:
            line = line.split(',')
            hs.append(correction*float(line[0]))
            hp.append(correction*float(line[2]))
            ts.append(correction*float(line[4]))
            tp.append(correction*float(line[6]))

            SIG_LEVEL = 0.01
            all_sig['hs'] = all_sig['hs'] and (float(line[1]) < SIG_LEVEL)
            some_sig['hs'] = some_sig['hs'] or (float(line[1]) < SIG_LEVEL)
            all_sig['hp'] = all_sig['hp'] and (float(line[3]) < SIG_LEVEL)
            some_sig['hp'] = some_sig['hp'] or (float(line[3]) < SIG_LEVEL)
            all_sig['ts'] = all_sig['ts'] and (float(line[5]) < SIG_LEVEL)
            some_sig['ts'] = some_sig['ts'] or (float(line[5]) < SIG_LEVEL)
            all_sig['tp'] = all_sig['tp'] and (float(line[7]) < SIG_LEVEL)
            some_sig['tp'] = some_sig['tp'] or (float(line[7]) < SIG_LEVEL)

        new_line = [f.replace('HUMOD',"H").replace("USR_TC_REL","TC-S").replace("Distr0", "Rand3750").replace("_wt1","").replace("_lim3750","")]
        for corr_name, corrs in zip(['hs', 'hp', 'ts', 'tp'], [hs, hp,ts,tp]):
            if len(corrs) == 0:
                new_line.append(f"-")
            else:
                symb = '*' if all_sig[corr_name] else 'Ankh ' if some_sig[corr_name] else ''
                if len(corrs) == 1:
                    new_line.append(f"{symb}{corrs[0]:.2f}")
                else:
                    new_line.append(f"{symb}{mean(corrs):.2f} ({stdev(corrs):.2f})")

        results.append(new_line)
    print(f"TARGET EPOCH {TARGET_EPOCH}")

    import copy
    tmp = copy.deepcopy(results[1:5])
    results[1:5] = results[5:9]
    results[5:9] = tmp

    print(tabulate(results))
    print(tabulate(results, tablefmt='latex'))



if __name__ == '__main__':
    main()
