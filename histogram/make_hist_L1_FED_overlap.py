import torch
from model.logistic_regression import LogisticRegression
import os
import matplotlib.pyplot as plt
from math import log10

if __name__ == '__main__':
    load_dir = 'exp/HUMOD_Distr0_bce_L1_wt1_lim3750'
    repeat = '0'
    epoch_target = 2

    net = LogisticRegression(768) # Model to get pred from BERT
    net.load_state_dict(torch.load(os.path.join(load_dir, repeat, f'HUM_chk_{epoch_target - 1}.pt')))
    weights = net.fc1.weight[0].detach().abs()
    weights_list = [log10(x) for x in weights.numpy().tolist()]

    net = LogisticRegression(768) # Model to get pred from BERT
    net.load_state_dict(torch.load(os.path.join('exp/HUMOD_IDK_bce_L1_wt1_lim3750', repeat, f'HUM_chk_{epoch_target - 1}.pt')))
    weights = net.fc1.weight[0].detach().abs()
    weights_list_idk = [log10(x) for x in weights.numpy().tolist()]

    #plt.figure(figsize=(8,6))
    plt.hist(weights_list_idk, bins=100, alpha=0.5, label="IDK")
    plt.hist(weights_list, bins=100, alpha=0.5, label="Ablated")
    plt.xlabel("log of weight magnitudes (log base 10)", size=14)
    #plt.ylabel("Count", size=14)
    plt.title("Log-Scale Histogram of Weights Learned by\nIDK and Ablated IDK (no sampling) on HUMOD")
    plt.legend(loc='upper right')
    plt.savefig("overlapping_histograms_IDK_ablate_humod.png")
