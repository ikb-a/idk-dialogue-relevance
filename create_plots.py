import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})
    for name, file in [('HUMOD', 'test_eval_exmple__HUMOD.tsv'),
                       ('P-DD', 'test_eval_exmple__P_DD.tsv'),
                       ('USR-TC', 'test_eval_exmple__USR_TC_REL.tsv'),
                       ('FED-COR', 'test_eval_exmple__FED_COR.tsv'),
                       ('FED-REL', 'test_eval_exmple__FED_REL.tsv')]:
        data = pd.read_csv(os.path.join('exp', 'HUMOD_IDK_bce_L1_wt1_lim3750', file),sep='\t')
        human = data['human']
        machine = data['HUMOD_IDK_bce_L1_wt1_lim3750']

        import seaborn as sns

        sns.regplot(x=human,
                    y=machine,
                    fit_reg=True,
                    x_jitter=0.1,
                    y_jitter=0,
                    scatter_kws={'alpha': 0.5})  # set transparency to 50%

        plt.title(f"Linearly Rescaled IDK vs Human ratings on the {name} test split", wrap=True)
        plt.xlabel("Human Rating", wrap=True)
        plt.ylabel("Linearly Rescaled IDK Score", wrap=True)
        plt.tight_layout()
        plt.savefig(f'{name}.jpg')
        plt.close()

