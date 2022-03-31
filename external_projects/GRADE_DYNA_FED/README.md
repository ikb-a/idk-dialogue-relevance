# Instructions

Checkout https://github.com/exe1023/DialEvalMetrics.git

goto commit f27d717  

https://github.com/exe1023/DialEvalMetrics/commit/f27d717cfb02b08ffd774e60faa6b319a766ae77

Follow repo instructions to setup GRADE/FED/DYNAEVAL checkpoints, and FED data.

Read instructions for GRADE, DYNAEVAL, and FED.

Next steps. Note, this section will be streamlined and revised. Note that at present, some steps may be omitted, or contained within the metric's specific instructions in the repo.

Create Python3.6 virtual env in `grade/env_grade` using dependencies in `grade_env.txt`; then follow GRADE repo instructions to install `grade/texar-pytorch/`

Create Python3.6 virtual env in `fed_server_env` using dependencies in `fed_server_env.txt`

Create `conda` virtual env using `gcn_environment.yml`

Copy over provided files, replacing as needed.

Create the softlink: `generate_IDK_data/data` that points to `dialogue-relevance/data/`

Update PATH and LD_LIBRARY_PATH paths in `eval_IDK_all.sh` using the location of your CUDA10.2 install

Run `generate_IDK_data/generate_data.py` to put data into format used by this repo.

Run `gen_IDK_all.sh` to get data in right formats for GRADE & DYNAEVAL. Need to update path to conda env.

Run FED metrics on fed data using `eval_metrics_FED.sh`

Run DynaEval on FED data using `eval_metrics_dynaeval.sh` (note need to update CUDA10.2 install paths)

Run GRADE on FED data using `eval_metrics_grade.sh (note, need to update conda path)

Read off GRADE & DYNAEVAL results using `read_IDK_all.sh` (note, need to update conda path)

Run `display_IDK_all.py`

