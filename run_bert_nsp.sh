#!/bin/bash

python3 bert_NSP.py USR_TC_REL
python3 bert_NSP.py HUMOD
python3 bert_NSP.py  --test USR_TC_REL
python3 bert_NSP.py --test HUMOD
python3 bert_NSP.py --test P_DD
python3 bert_NSP.py --test FED_COR
python3 bert_NSP.py --test FED_REL
