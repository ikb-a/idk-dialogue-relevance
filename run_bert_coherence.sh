#!/bin/bash

python3 bert_coherence.py USR_TC_REL
python3 bert_coherence.py HUMOD
python3 max_bert_coherence.py USR_TC_REL
python3 max_bert_coherence.py HUMOD

python3 bert_coherence.py  --test USR_TC_REL
python3 bert_coherence.py --test HUMOD
python3 bert_coherence.py --test P_DD
python3 bert_coherence.py --test FED_COR
python3 bert_coherence.py --test FED_REL
python3 max_bert_coherence.py --test USR_TC_REL
python3 max_bert_coherence.py --test HUMOD
python3 max_bert_coherence.py --test P_DD
python3 max_bert_coherence.py --test FED_COR
python3 max_bert_coherence.py --test FED_REL
