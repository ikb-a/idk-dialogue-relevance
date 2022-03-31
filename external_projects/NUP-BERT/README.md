## Instructions

Checkout https://github.com/vitouphy/usl_dialogue_metric.git commit 2b2037e (i.e., https://github.com/vitouphy/usl_dialogue_metric/commit/2b2037e4d76a08925e47c506f1edc663072161e2 )

Note all code in this folder is modified from the above.

resulting models will be saved under `exp/???/lightning_logs/...` and .json file of predicted NUP scores in `exp/???/blah.json`

Again, copy the provided files, replacing as needed, and install the 
requirements we've provided. Note, I had to use Python3.6 to get this code to work,
and you may have to update pip first.

Next, create softlink to `data/lines` inside `usl_scores` subdirectory.

Next, change into the `usl_score` directory, and run `train_all.sh` to 
train the models.

Finally run `python3 -u eval_test.py | tee -i eval_test_log.txt` to get
test results; then run `python3 -u eval_val_all.py | tee -i eval_val_log.txt` to get
validation results.

## Notes

Need to run `eval_test.py` and save/tee the printed output

Note that `eval_test.py` and `eval_val_all.py` both contain a dictionary named
`models` that contains the lightning version numbers of the models created.
These numbers may have to be adjusted manually if you run models before
`train_all.sh` is run, or if you interrupt and resume the script. Just 
look for the version numbers in `exp_final/X/lightning_logs/version_#` and map
`X` in `models` to all corresponding version numbers.
