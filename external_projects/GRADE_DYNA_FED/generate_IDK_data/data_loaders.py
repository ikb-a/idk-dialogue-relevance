# TODO: one data loader for each dataset; all the same, but needed due to structure of
# gen_data.py

import json
import os

def load_IDK_humod_test_data(basedir):
    with open(os.path.join(basedir, "IDK_humod_test.json"), 'r') as infile:
        return json.load(infile)


def load_IDK_humod_val_data(basedir):
    with open(os.path.join(basedir, "IDK_humod_val.json"), 'r') as infile:
        return json.load(infile)


def load_IDK_usr_tc_test_data(basedir):
    with open(os.path.join(basedir, "IDK_usr_tc_test.json"), 'r') as infile:
        return json.load(infile)


def load_IDK_usr_tc_val_data(basedir):
    with open(os.path.join(basedir, "IDK_usr_tc_val.json"), 'r') as infile:
        return json.load(infile)


def load_IDK_fed_cor_data(basedir):
    with open(os.path.join(basedir, "IDK_fed_cor.json"), 'r') as infile:
        return json.load(infile)


def load_IDK_fed_rel_data(basedir):
    with open(os.path.join(basedir, "IDK_fed_rel.json"), 'r') as infile:
        return json.load(infile)


def load_IDK_pang_dd_test_data(basedir):
    with open(os.path.join(basedir, "IDK_pang_dd_test.json"), 'r') as infile:
        return json.load(infile)


