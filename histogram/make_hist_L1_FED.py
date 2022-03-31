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
    weights_list = weights.numpy().tolist()
    plt.hist(weights_list)
    plt.savefig('humod_l1_bce_weights.jpg')
    plt.close()
    plt.hist([log10(x) for x in weights_list])
    plt.xlabel('log of weight magnitudes (log base 10)')
    plt.title('Log-Scale Histogram of Weights Learned by \nAblated IDK (random sampling) on HUMOD')
    plt.savefig('humod_l1_bce_weights_log10.jpg')

    weights_list.sort(reverse=True)
    print(weights_list[:10])

