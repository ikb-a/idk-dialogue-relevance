import torch
from transformers import BertModel
from model.logistic_regression import LogisticRegression
from scipy.stats import pearsonr, spearmanr
import numpy as np
from data_loader.test_data_loader import DataLoaderGenerator, LoaderType, QualDataset
import os
from transformers import BertTokenizer


"""
NOTE: For USR-TC this evaluates everything in the test set (human & non-human)
Just eval first repetition
"""

MAX_LEN = 512
BATCHSIZE = 32



def validate(data_loader_generator, bert_model, net, outfile, tokenizer, device='cuda'):
    with torch.no_grad():
        valloader = data_loader_generator.get_val_data_loader()
        val_HUM_Y = data_loader_generator.get_val_y()
        predictions = []
        dials = []

        j = 0

        for i, data in enumerate(valloader, 0):
            data = [x.to(torch.int64).to(device) for x in data]
            tok, att, ty = data
            _, feats = bert_model(input_ids=tok, attention_mask=att, token_type_ids = ty)
            val_pred = net(feats)
            val_pred = val_pred.detach().cpu().numpy()
            val_pred = np.squeeze(val_pred)
            val_pred = val_pred.tolist()
            predictions += val_pred
            for t, vp in zip(tok, val_pred):
                convo = tokenizer.convert_ids_to_tokens(t)
                convo = ' '.join([c for c in convo if c != '[PAD]'])
                dials.append(convo)

        return dials, val_HUM_Y, predictions

def main(device='cuda', epoch_target=2):
    print("Loading BERT")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)
    torch.cuda.empty_cache()

    net = LogisticRegression(768) # Model to get pred from BERT

    print("Start testing")
    for test_dataset in ["HUMOD", 'USR_TC_REL', 'P_DD', 'FED_REL', 'FED_COR']:
        data_loader_generator = DataLoaderGenerator(bert_max_len=MAX_LEN,
                                                    to_load=QualDataset[test_dataset],
                                                    load_type=LoaderType['IDK'],
                                                    batchsize=BATCHSIZE,
                                                    skip_tr=True)
        # Save all domain performances
        net.eval()

        LOADERS = ['IDK']
        LOSSES = ["bce"]
        REGS = ["L1"]
        LIMIT = 3750 #10

        for dataset in ["HUMOD"]:
            for loader in LOADERS:
                for loss in LOSSES:
                    for reg_nme in REGS:

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

                        target_file = save_dir+f'/test_eval_exmple_{file_num}_{test_dataset}.tsv'

                        if os.path.exists(target_file):
                            print(f"SKIPPING existing: test {test_dataset}; {dataset} {loader} {loss} {reg_nme}")
                        else:
                            with open(target_file, 'w') as outfile:
                                print(f'"dialogue"\t"human"\t"{save_dir.split("/")[1]}"', file=outfile)

                            print(f"test {test_dataset}; {dataset} {loader} {loss} {reg_nme}")

                            # Just look at first model; i=0
                            i=0

                            net.load_state_dict(torch.load(os.path.join(save_dir, str(i), f'HUM_chk_{epoch_target - 1}.pt')))
                            net.to(device)
                            net.eval()

                            with open(target_file, 'a') as outfile:
                                dials, human, machine = validate(data_loader_generator, bert_model, net, outfile, tokenizer)


                                machine = np.array(machine)
                                machine = 1 - machine  # Adjust for backwards sigmoid
                                machine -= min(machine)

                                scale = 2 if test_dataset == 'USR_TC_REL' or 'FED_' in test_dataset else 4
                                print(scale)

                                machine = 1 + (machine/max(machine))*scale

                                for convo, vh, vp in zip(dials, human, machine):
                                    print(f"{convo}\t{vh:.2f}\t{vp:.2f}", file=outfile)


if __name__ == '__main__':
    main(epoch_target=2)
