import torch
from transformers import BertModel
from model.logistic_regression import triplet_loss, triplet_loss2, LogisticRegression
from scipy.stats import pearsonr, spearmanr
import numpy as np
from data_loader.data_loader import DataLoaderGenerator, LoaderType, QualDataset
import os
import argparse
import datetime
from tqdm import tqdm

BATCHSIZE = 6
MARGIN = 0.4 #0.5
MAX_LEN = 512

EPOCHS = 2
LR = 0.001

LOSS_FUNC = triplet_loss2



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="What dataset to load", choices=['HUMOD', 'USR_TC_REL'])
    parser.add_argument("loader", help="What loader to use", choices=['IDK', 'Rand3750', 'ICS', 'OK'])
    parser.add_argument("loss", help="loss function", choices=["trip2", "bce"], default="trip2")
    parser.add_argument('--margin', help='Margin Hyperparameter', default=0.4, type=float)
    parser.add_argument('--reg_nme', help='Regularizer name', default=None, choices=['L1', 'L2'])
    parser.add_argument('--reg_wt', help='Regularizer name', default=1, type=float)
    parser.add_argument('--limit', help='Put a limit on train/val size', default=None, type=int)
    args = parser.parse_args()
    print(args)
    run(args.dataset, args.loader, args.loss, args.margin, args.reg_nme, args.reg_wt, args.limit)

def run(argsdataset, argsloader, argsloss, argsmargin=0.4,
        argsreg_nme=None, argsreg_wt=1, argslimit=None, repeat_idx=None):
    DATASET = QualDataset[argsdataset]

    # For HUMOD, use provided random distractors
    if argsdataset == "HUMOD" and argsloader == "Rand3750":
        argsloader="Distr0"

    LOADER = LoaderType[argsloader]
    MARGIN = argsmargin
    REG_NAME = argsreg_nme
    REG_WT = argsreg_wt

    if argsloss == "trip2":
        LOSS_FUNC = triplet_loss2
    elif argsloss == 'bce':
        def bce(pos_score, neg_score, margin, device):
            # for consistency with triplet loss, I have positive & negative
            # backwards (i.e., positive should be 0)
            return -1 * torch.log(neg_score).sum() - torch.log(1-pos_score).sum()
        LOSS_FUNC = bce

    if MARGIN == 0.4:
        save_dir = 'exp/'+DATASET.name+'_'+LOADER.name
    else:
        save_dir = 'exp/'+DATASET.name + '_' + LOADER.name + f'_M{MARGIN}'

    if argsloss != "trip2":
        save_dir += '_' + argsloss

    if REG_NAME is not None:
        save_dir += '_' + REG_NAME + f'_wt{REG_WT}'

    if argslimit is not None:
        save_dir += f'_lim{argslimit}'

    if repeat_idx is not None:
        root_save_dir = save_dir
        save_dir = f"{save_dir}/{repeat_idx}"

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_dir + f'/HUM_chk_{EPOCHS - 1}.pt'):
        print("Fully trained checkpoint already exists, terminating")
        return

    dev = "cuda" #cuda
    device = torch.device(dev)
    print("Saving config")

    with open(save_dir + '/config.txt', 'w') as outfile:
        print(f'CONFIG\nBatchsize: {BATCHSIZE}'
              f'\nMargin: {MARGIN}'
              f'\nMAX_LEN: {MAX_LEN}'
              f'\nEPOCHS: {EPOCHS}'
              f'\nLR: {LR}'
              f'\nLOSS FUNC: {LOSS_FUNC}'
              f'REG_NAME: {REG_NAME}'
              f'REG_WT: {REG_WT}'
              f'\nLIMIT: {argslimit}', file=outfile)

    print("Loading training data")
    data_loader_generator = DataLoaderGenerator(bert_max_len=MAX_LEN, to_load=DATASET, load_type=LOADER, batchsize=BATCHSIZE, limit=argslimit)
    print("Creating Val Loader")
    valloader = data_loader_generator.get_val_data_loader()
    print("Getting Val Scores")
    val_HUM_Y = data_loader_generator.get_val_y()

    print("Loading Model")

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model.to(device)
    torch.cuda.empty_cache()

    net = LogisticRegression(768) # Model to get pred from BERT

    with open(save_dir+'/log.txt', 'a') as logfile:
        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=logfile)

        if REG_NAME is not None:
            print(f"Using {REG_NAME} regularization.")
            print(f"Using {REG_NAME} regularization.", file=logfile)

        first_epoch = 0
        most_recent_checkpoint = -1

        # Check if any checkpoints exist
        for entry in os.scandir(save_dir):
            if entry.name.startswith('HUM_chk_') and entry.name.endswith('.pt'):
                checkpoint_number = int(entry.name[8:-3])
                if most_recent_checkpoint < checkpoint_number:
                    most_recent_checkpoint = checkpoint_number

        # Load & update epoch counter if applicable.
        if most_recent_checkpoint > -1:
            net.load_state_dict(torch.load(save_dir + f'/HUM_chk_{most_recent_checkpoint}.pt'))
            print(f'Loading from Checkpoint /HUM_chk_{most_recent_checkpoint}.pt')
            print(f'Loading from Checkpoint /HUM_chk_{most_recent_checkpoint}.pt', file=logfile)
            first_epoch = most_recent_checkpoint + 1

        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=LR)

        print("Starting Training")
        print("Starting Training", file=logfile)

        for epoch in range(first_epoch, EPOCHS):  # loop over the dataset multiple times

            running_loss = 0.0
            print("Loading train loader")
            print("Loading train loader", file=logfile)
            trainloader = data_loader_generator.get_train_loader()
            print(f"Starting Epoch {epoch}")
            print(f"Starting Epoch {epoch}", file=logfile)
            for i, data in enumerate(tqdm(trainloader)):
                with torch.no_grad():
                    #print([x.shape for x in data])
                    data = [x.squeeze().to(device) for x in data]
                    #print([x.shape for x in data])

                    if len(list(data[0].shape)) == 1:
                        data = [x.unsqueeze(0) for x in data]

                    tok_x1, att_x1, ty_x1, tok_x2, att_x2, ty_x2 = data
                    #del data

                    _, feats1 = bert_model(input_ids=tok_x1, attention_mask = att_x1, token_type_ids=ty_x1)
                    feats1.detach() # don't backprop into pretrained BERT model
                    #print(feats1.shape)
                    #del tok_x1
                    #del att_x1
                    #del ty_x1
                    #torch.cuda.empty_cache()

                    _, feats2 = bert_model(input_ids=tok_x2, attention_mask = att_x2, token_type_ids=ty_x2)
                    feats2.detach() # don't backprop into pretrained BERT model
                    #print(feats2.shape)
                    #del tok_x2
                    #del att_x2
                    #del ty_x2
                    #torch.cuda.empty_cache()
                optimizer.zero_grad()

                pos_score = net(feats1).flatten()
                neg_score = net(feats2).flatten()
                loss = LOSS_FUNC(pos_score, neg_score, margin=MARGIN, device=device)

                if REG_NAME == 'L1':
                    loss += REG_WT*torch.norm(net.fc1.weight.flatten(), p=1)
                elif REG_NAME == 'L2':
                    loss += REG_WT*(torch.norm(net.fc1.weight.flatten(), p=2)**2)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                ITEMS = 600
                if i % ITEMS == ITEMS - 1:
                    print('[%d, %5d] loss: %.6f' %
                          (epoch + 1, i + 1, running_loss / ITEMS))
                    print('[%d, %5d] loss: %.6f' %
                          (epoch + 1, i + 1, running_loss / ITEMS), file=logfile)
                    running_loss = 0.0

            torch.save(net.state_dict(), "./" + save_dir + "/HUM_chk_"+str(epoch)+".pt")

            print(datetime.datetime.now())
            print(datetime.datetime.now(), file=logfile)

            # Get validation loss
            print(str.format("EPOCH {}",epoch + 1))
            print(str.format("EPOCH {}",epoch + 1), file=logfile)
            with torch.no_grad():
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
                print("Pearson Rho: {} p={}".format(coef1, p1))
                print("Pearson Rho: {} p={}".format(coef1, p1), file=logfile)

                coef2, p2 = spearmanr(predictions, val_HUM_Y)
                print("Spearman Tau: {} p={}".format(coef2, p2))
                print("Spearman Tau: {} p={}".format(coef2, p2), file=logfile)

                with open(save_dir + "/val_corr.txt", 'a') as outfile:
                    print(str.format("EPOCH {}", epoch + 1), file=outfile)
                    print("Pearson Rho: {} p={}".format(coef1, p1), file=outfile)
                    print("Spearman Tau: {} p={}".format(coef2, p2), file=outfile)

            print(datetime.datetime.now())
            print(datetime.datetime.now(), file=logfile)

            # Save all domain performances
            if repeat_idx is not None and epoch in [1,3,4]:  # i.e., epoch 2, 4 or 5
                def validate(dataset):
                    net.eval()
                    with torch.no_grad():
                        print("Loading validation data")
                        data_loader_generator = DataLoaderGenerator(bert_max_len=MAX_LEN, to_load=QualDataset[dataset], load_type=LOADER, batchsize=BATCHSIZE, skip_tr=True, limit=argslimit)
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

                if epoch == 1:
                    target_file_name = "/results.csv"
                else:
                    target_file_name = f'/results{epoch+1}.csv'
                if not os.path.exists(root_save_dir+target_file_name):
                    with open(root_save_dir+target_file_name, 'w') as outfile:
                        print('"spearman_h","spear_sig_h","pearson_h","pear_sig_h",'
                              '"spearman_tc","spear_sig_tc","pearson_tc","pear_sig_tc"', file=outfile)
                with open(root_save_dir+target_file_name, 'a') as outfile:
                    for val_dataset in ["HUMOD", "USR_TC_REL"]:
                        coef1, p1, coef2, p2 = validate(val_dataset)
                        print(f"{coef2},{p2},{coef1},{p1}", file=outfile, end='')
                        if val_dataset != "USR_TC_REL":
                            print(',',file=outfile,end='')
                    print('\n',file=outfile, end='')

        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=logfile)



if __name__ == '__main__':
    main()
