from unsup_bert import run
import os

if __name__ == '__main__':
    REPS = 3
    DATASETS = ['USR_TC_REL', 'HUMOD']
    LOADERS = ['IDK', 'Rand3750']
    LOSSES = ["trip2", "bce"]
    REGS = ["L1", "L2", None]
    LIMIT = 3750 #10

    EPOCH_TARGET = 2

    for i in range(REPS):
        for dataset in DATASETS:
            for loader in LOADERS:
                for loss in LOSSES:
                    for reg_nme in REGS:
                        # Humod gets random distractor from HUMOD
                        if dataset == 'HUMOD' and loader == 'Rand3750':
                            loader = 'Distr0'

                        save_dir = 'exp/'+dataset+'_'+loader
                        if loss != "trip2":
                            save_dir += '_' + loss
                        if reg_nme is not None:
                            save_dir += '_' + reg_nme + f'_wt1'
                        save_dir += f'_lim{LIMIT}'

                        file_num = EPOCH_TARGET if EPOCH_TARGET != 2 else ''
                        if os.path.exists(save_dir+f'/results{file_num}.csv'):
                            with open(save_dir+f'/results{file_num}.csv', 'r') as infile:
                                tmp = infile.readlines()
                            tmp = [t for t in tmp if t.strip() != ""]
                            if len(tmp) > i + 1:
                                print(f"SKIP EXISTING {dataset} {loader} {loss} {reg_nme} REP: {i}")
                                continue
                        # Run experiment
                        print('=========================================')
                        print(f"{dataset} {loader} {loss} {reg_nme} REP: {i}")
                        run(argsdataset=dataset, argsloader=loader, argsloss=loss, argsreg_nme=reg_nme,argslimit=LIMIT, repeat_idx=i)



    # Follup ICS experiment
    REPS = 3
    DATASETS = ['USR_TC_REL', 'HUMOD']
    LOADERS = ['ICS', 'OK']
    LOSSES = ["bce"]
    REGS = ["L1"]
    LIMIT = 3750 #10

    EPOCH_TARGET = 2

    for i in range(REPS):
        for dataset in DATASETS:
            for loader in LOADERS:
                for loss in LOSSES:
                    for reg_nme in REGS:
                        save_dir = 'exp/'+dataset+'_'+loader
                        if loss != "trip2":
                            save_dir += '_' + loss
                        if reg_nme is not None:
                            save_dir += '_' + reg_nme + f'_wt1'
                        save_dir += f'_lim{LIMIT}'

                        file_num = EPOCH_TARGET if EPOCH_TARGET != 2 else ''
                        if os.path.exists(save_dir+f'/results{file_num}.csv'):
                            with open(save_dir+f'/results{file_num}.csv', 'r') as infile:
                                tmp = infile.readlines()
                            tmp = [t for t in tmp if t.strip() != ""]
                            if len(tmp) > i + 1:
                                print(f"SKIP EXISTING {dataset} {loader} {loss} {reg_nme} REP: {i}")
                                continue
                        # Run experiment
                        print('=========================================')
                        print(f"{dataset} {loader} {loss} {reg_nme} REP: {i}")
                        run(argsdataset=dataset, argsloader=loader, argsloss=loss, argsreg_nme=reg_nme,argslimit=LIMIT, repeat_idx=i)
