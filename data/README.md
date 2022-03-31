#Data

download into this folder the following datasets. Next go to 
repo's root directory and run preprocess.py; this will create
subdirectories for each dataset.

NOTE: To run other author's metrics, next run csv_ify.py (will create
a `csv` folder that's the requried format for GPT-2 adjusted conditional
probability). Note that you must run from the same directory, but with
python path being the root above (i.e., run in PyCharm or use 
`export PYTHONPATH=".." \ python3 csv_ify.py` from terminal). 
Then run each of the scripts in a subdirectory of `lines` to produce format
required for USL-H (i.e., BERT-NUP), and rics dialogue eval (i.e., COS-FT) 

## HUMOD

HUMOD dataset taken from https://github.com/erincmer/HUMOD

Link to paper: "Human Annotated Dialogues Dataset for Natural Conversational Agents"
https://www.mdpi.com/2076-3417/10/3/762/htm

The dataset (`HUMOD_Dataset.TXT`) is available for download at https://github.com/erincmer/HUMOD/blob/master/HUMOD_Dataset.TXT

We use the first 3750 elements as the test split, the next 500 as valid,
and the last 500 as test.

```
@Article{app10030762,
AUTHOR = {Merdivan, Erinc and Singh, Deepika and Hanke, Sten and Kropf, Johannes and Holzinger, Andreas and Geist, Matthieu},
TITLE = {Human Annotated Dialogues Dataset for Natural Conversational Agents},
JOURNAL = {Applied Sciences},
VOLUME = {10},
YEAR = {2020},
NUMBER = {3},
ARTICLE-NUMBER = {762},
URL = {https://www.mdpi.com/2076-3417/10/3/762},
ISSN = {2076-3417},
ABSTRACT = {Conversational agents are gaining huge popularity in industrial applications such as digital assistants, chatbots, and particularly systems for natural language understanding (NLU). However, a major drawback is the unavailability of a common metric to evaluate the replies against human judgement for conversational agents. In this paper, we develop a benchmark dataset with human annotations and diverse replies that can be used to develop such metric for conversational agents. The paper introduces a high-quality human annotated movie dialogue dataset, HUMOD, that is developed from the Cornell movie dialogues dataset. This new dataset comprises 28,500 human responses from 9500 multi-turn dialogue history-reply pairs. Human responses include: (i) ratings of the dialogue reply in relevance to the dialogue history; and (ii) unique dialogue replies for each dialogue history from the users. Such unique dialogue replies enable researchers in evaluating their models against six unique human responses for each given history. Detailed analysis on how dialogues are structured and human perception on dialogue score in comparison with existing models are also presented.},
DOI = {10.3390/app10030762}
}
```

## USR-TC

USR-TC Data set (specifically the file  tc_usr_data.json ) taken from http://shikib.com/usr, USR: An Unsupervised and Reference Free Evaluation Metric for Dialog (Mehri and Eskenazi, 2020)

http://shikib.com/tc_usr_data.json


USR-TC is annotated Amazon Topical-Chat data (from 'frequent test set').

We divide the dataset into two halves, first 30 elements are valid,
last 30 elements are test.

```
@inproceedings{DBLP:conf/acl/MehriE20,
  author    = {Shikib Mehri and
               Maxine Esk{\'{e}}nazi},
  editor    = {Dan Jurafsky and
               Joyce Chai and
               Natalie Schluter and
               Joel R. Tetreault},
  title     = {{USR:} An Unsupervised and Reference Free Evaluation Metric for Dialog
               Generation},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2020, Online, July 5-10, 2020},
  pages     = {681--707},
  publisher = {Association for Computational Linguistics},
  year      = {2020},
  url       = {https://doi.org/10.18653/v1/2020.acl-main.64},
  doi       = {10.18653/v1/2020.acl-main.64},
  timestamp = {Fri, 06 Aug 2021 00:41:00 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/MehriE20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Topical Chat

One of the datasets annotated in USR. As the USR data came from
the Topical Chat test set, we can safely use the TC train and 
validation sets.

For consistency with USR TC data, it is converted to lowercase,
and word-tokenized.

Amazon Topical Chat data from https://github.com/alexa/Topical-Chat/tree/master/conversations

Download the files `train.json`, `valid_freq.json`, and `valid_rare.json`

Turn ratings are:
"turn_rating": ""
"turn_rating": "Excellent"
"turn_rating": "Good"
"turn_rating": "Passable"
"turn_rating": "Not Good"
"turn_rating": "Poor"

```
@inproceedings{Gopalakrishnan2019,
author={Karthik Gopalakrishnan and Behnam Hedayatnia and Qinlang Chen and Anna Gottardi and Sanjeev Kwatra and Anu Venkatesh and Raefer Gabriel and Dilek Hakkani-Tür},
title={{Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations}},
year=2019,
booktitle={Proc. Interspeech 2019},
pages={1891--1895},
doi={10.21437/Interspeech.2019-3079},
url={http://dx.doi.org/10.21437/Interspeech.2019-3079}
}
```

## Pang et al. DailyDilogue

Annotated subset of DailyDialogue dataset, from the same paper
as the NORM-PROB metric.

https://github.com/alexzhou907/dialogue_evaluation/blob/d7f08e0e1d2ce9cdd422b3a6c6c02adccfbfea27/context_data_release.csv

```
@inproceedings{pang-etal-2020-towards,
    title = "Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation",
    author = "Pang, Bo  and
      Nijkamp, Erik  and
      Han, Wenjuan  and
      Zhou, Linqi  and
      Liu, Yixian  and
      Tu, Kewei",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.333",
    doi = "10.18653/v1/2020.acl-main.333",
    pages = "3619--3629",
}
```

## FED

FED Dataset.

http://shikib.com/fed_data.json

```
@inproceedings{mehri-eskenazi-2020-unsupervised,
    title = "Unsupervised Evaluation of Interactive Dialog with {D}ialo{GPT}",
    author = "Mehri, Shikib  and
      Eskenazi, Maxine",
    booktitle = "Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.sigdial-1.28",
    pages = "225--235",
    abstract = "It is important to define meaningful and interpretable automatic evaluation metrics for open-domain dialog research. Standard language generation metrics have been shown to be ineffective for dialog. This paper introduces the FED metric (fine-grained evaluation of dialog), an automatic evaluation metric which uses DialoGPT, without any fine-tuning or supervision. It also introduces the FED dataset which is constructed by annotating a set of human-system and human-human conversations with eighteen fine-grained dialog qualities. The FED metric (1) does not rely on a ground-truth response, (2) does not require training data and (3) measures fine-grained dialog qualities at both the turn and whole dialog levels. FED attains moderate to strong correlation with human judgement at both levels.",
}

```
