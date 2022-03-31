import os
from data_preprocess.humod_preproc import main as humod_main
from data_preprocess.humod_split import main as humod_split
from data_preprocess.usr_preproc_split import main as usr_split
from data_preprocess.tc_preproc import main as tc_split
from data_preprocess.pang_dd_preproc import main as p_dd_make
from data_preprocess.fed_preproc import main as fed_main

"""
Preprocess all data.

Small corrections to typos in HUMOD data.

Undo contraction splitting in USR TC data.
"""

if __name__ == '__main__':
    # Preprocess HUMOD into corpus object
    if not os.path.exists('data/HUMOD_Dataset2.TXT') or not os.path.exists('data/HUMOD_v2.0.json'):
        humod_main()

    if not os.path.exists('data/humod'):
        humod_split()

    if not os.path.exists('data/usr_tc'):
        usr_split()

    tc_split()

    if not os.path.exists('data/pang_dd'):
        p_dd_make()

    if not os.path.exists('data/fed'):
        fed_main()
