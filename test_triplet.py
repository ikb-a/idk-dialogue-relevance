import torch
from transformers import BertModel
from model.logistic_regression import LogisticRegression
from scipy.stats import pearsonr, spearmanr
import numpy as np
from data_loader.test_data_loader import DataLoaderGenerator, LoaderType, QualDataset
import os

"""NOTE: For USR-TC this evaluates everything in the test set (human & non-human)"""

MAX_LEN = 512
BATCHSIZE = 32
REPS = 3

def validate(data_loader_generator, bert_model, net, device='cuda'):
    with torch.no_grad():
        valloader = data_loader_generator.get_val_data_loader()
        val_HUM_Y = data_loader_generator.get_val_y()
        predictions = []

        for i, data in enumerate(valloader, 0):
            data = [x.to(torch.int64).to(device) for x in data]
            tok, att, ty = data
            _, feats = bert_model(input_ids=tok, attention_mask=att, token_type_ids = ty)
            val_pred = net(feats)
            val_pred = val_pred.detach().cpu().numpy()
            val_pred = np.squeeze(val_pred)
            val_pred = val_pred.tolist()
            predictions += val_pred
        coef1, p1 = pearsonr(predictions, val_HUM_Y)
        coef2, p2 = spearmanr(predictions, val_HUM_Y)
        return coef1, p1, coef2, p2


def main(device='cuda', epoch_target=2):
    print("Loading BERT")

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)
    torch.cuda.empty_cache()

    net = LogisticRegression(768) # Model to get pred from BERT

    print("Start testing")
    for test_dataset in ["HUMOD", "USR_TC_REL", "P_DD", "FED_COR", "FED_REL"]:
        print("Loading validation data")
        data_loader_generator = DataLoaderGenerator(bert_max_len=MAX_LEN,
                                                    to_load=QualDataset[test_dataset],
                                                    load_type=LoaderType['IDK'],
                                                    batchsize=BATCHSIZE,
                                                    skip_tr=True)
        # Save all domain performances
        net.eval()

        LOADERS = ['IDK', 'Rand3750', 'ICS', 'OK']
        LOSSES = ["trip2", "bce"]
        REGS = ["L1", "L2", None]
        LIMIT = 3750 #10

        for dataset in ["HUMOD", "USR_TC_REL"]:
            for loader in LOADERS:
                for loss in LOSSES:
                    for reg_nme in REGS:
                        # Skip settings that were not tested
                        if loader in ['ICS', 'OK'] and (reg_nme != "L1" or loss != "bce"):
                            continue

                        # For HUMOD, use provided random distractors
                        if dataset == "HUMOD" and loader == "Rand3750":
                            loader="Distr0"

                        save_dir = 'exp/'+dataset+'_'+loader
                        if loss != "trip2":
                            save_dir += '_' + loss
                        if reg_nme is not None:
                            save_dir += '_' + reg_nme + f'_wt1'
                        save_dir += f'_lim{LIMIT}'

                        file_num = epoch_target if epoch_target != 2 else ''

                        target_file = save_dir+f'/test_results{file_num}_{test_dataset}.csv'

                        if os.path.exists(target_file):
                            print(f"SKIPPING existing: test {test_dataset}; {dataset} {loader} {loss} {reg_nme}")
                        else:
                            with open(target_file, 'w') as outfile:
                                print('"spearman","spear_sig","pearson","pear_sig"', file=outfile)

                            print(f"test {test_dataset}; {dataset} {loader} {loss} {reg_nme}")
                            for i in range(REPS):

                                net.load_state_dict(torch.load(os.path.join(save_dir, str(i), f'HUM_chk_{epoch_target - 1}.pt')))
                                net.to(device)
                                net.eval()

                                with open(target_file, 'a') as outfile:
                                    coef1, p1, coef2, p2 = validate(data_loader_generator, bert_model, net)
                                    print(f"{coef2},{p2},{coef1},{p1}", file=outfile)
                                    print(f"Spear: {coef2},{p2}, Pear: {coef1},{p1}")


if __name__ == '__main__':
    main(epoch_target=2)
