## Instructions

Checkout https://github.com/ricsinaruto/dialog-eval.git

Note all code in this folder modified from the above.

goto commit af95efc (i.e., https://github.com/ricsinaruto/dialog-eval/commit/af95efcdcea8499b1c48a909aae1d8643efabe21)

Softlink `lines` directory to root of checked out project. 

Copy over the provided files, replacing as needed.

To reproduce our test results, run `run_test.sh`. 
Say yes when the script asks if you want to use FastText embeddings.
The results will be
saved at `lines/XXX/test/corr_scores_coherence.txt`

To run on validation split, repeat, but replacing `test` in the filenames with `valid`
