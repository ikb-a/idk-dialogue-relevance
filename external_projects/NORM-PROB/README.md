
## Instructions

clone https://github.com/alexzhou907/dialogue_evaluation.git

NOTE: All code in this folder is modified from the above

goto commit `d7f08e0` (i.e., https://github.com/alexzhou907/dialogue_evaluation/commit/d7f08e0e1d2ce9cdd422b3a6c6c02adccfbfea27 )

The authors' dataset `context_data_release.csv` (i.e., P-DD) already 
should exist in root directory. Create softlink
to `csv` folder in root. 

Copy all files here into the repo, replacing as needed (install 
requirements from the requirements provided by us).

Run `metrics_evaluation.py` with following arguments:

```
python3 metrics_evaluation.py --file-path='csv/usr_tc_rel_val.csv'
```

Results will be appended to 'results.txt' file.

Full list of Namespaces for our experiments:

```
# Validation
Namespace(file_path='csv/usr_tc_rel_val.csv', metric='context', ngram=2, output_file_path='tc.csv', pretrained_model_path='gpt2')
Namespace(file_path='csv/val_HUMOD.csv', metric='context', ngram=2, output_file_path='humod.csv', pretrained_model_path='gpt2')
# Testing
#NOTE: Immediately below is P-DD
Namespace(file_path='context_data_release.csv', metric='context', ngram=2, output_file_path='test_output_context.csv', pretrained_model_path='gpt2')
Namespace(file_path='csv/test_HUMOD.csv', metric='context', ngram=2, output_file_path='test_output_context.csv', pretrained_model_path='gpt2')
Namespace(file_path='csv/usr_tc_rel_test.csv', metric='context', ngram=2, output_file_path='test_output_context.csv', pretrained_model_path='gpt2')
Namespace(file_path='csv/fed_cor.csv', metric='context', ngram=2, output_file_path='test_output_context.csv', pretrained_model_path='gpt2')
Namespace(file_path='csv/fed_rel.csv', metric='context', ngram=2, output_file_path='test_output_context.csv', pretrained_model_path='gpt2')
```

## Requirements & Bugfix

Note that the author's recommended requirements were:

```
python == 3.7.6
pytorch_transformers  == 1.2.0
pytorch_pretrained_bert == 0.6.2
tensorboardX  == 1.9
matplotlib == 3.1.1
nltk == 3.4.5
numpy == 1.16.4
pandas == 0.25.3
scipy == 1.3.2
seaborn == 0.9.0
scikit-image == 0.15.0
torch == 1.5.0
tqdm == 4.42.1
```

NOTE: I used python3.7, and the above for transformer don't work anymore (update broke
unicode support)

I fixed it by adjusting line 224 of env/lib/python3.7/site-packages/pytorch_pretrained_bert/tokenization_gpt2.py
using the fix here: https://github.com/huggingface/transformers/issues/537

You may also need to update pip to be able to install the dependencies.
