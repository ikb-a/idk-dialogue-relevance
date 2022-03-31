"""
Data loader for unsupervised BERT experiments. Specifically, converts Corpus objects produced by code in Data/Quality
into training & validation sets for the unsupervised experiments.
"""

from typing import Tuple, List, Optional
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from utils.CorpusRepr import load_corpus
from torch.utils.data import Dataset
from enum import Enum
import copy
import random
import torch

random.seed(42)


class MyDataset(Dataset):
    datax1 = []
    datax2 = []

    def __init__(self, datax1, datax2):
        assert(len(datax1) == len(datax2))
        self.datax1 = datax1
        self.datax2 = datax2

    def __len__(self):
        return len(self.datax1)

    def __getitem__(self, item):
        return self.datax1[item]['input_ids'], self.datax1[item]['attention_mask'], self.datax1[item]['token_type_ids'],\
               self.datax2[item]['input_ids'], self.datax2[item]['attention_mask'], self.datax2[item]['token_type_ids']


class MyDataset2(Dataset):
    datax1 = []
    datay = []

    def __init__(self, datax1, datay):
        assert(len(datax1) == len(datay))
        self.datax1 = datax1
        self.datay = datay

    def __len__(self):
        return len(self.datax1)

    def __getitem__(self, item):
        return self.datax1[item]['input_ids'], self.datax1[item]['attention_mask'], self.datax1[item]['token_type_ids'],\
               self.datay[item]


class QualDataset(Enum):
    HUMOD = 6
    USR_TC_REL = 9


class LoaderType(Enum):
    ShfleTr = 1  # Shuffle gold responses from training data to use as distractors
    ShfleTrPerEp = 2 # Use a different shuffling of distractors every epoch.
    Distr0 = 3  # use the zeroeth distractor
    IDK = 4  # Use "i don't know" as the distractor
    Rand3750 = 5  # Take first 3750 entries of training data, and add random distractors from rest
                  # Fix seed.
    ICS = 6
    OK=7


ROOT = 'data/'
ROOT2 = 'data/'
# Map enum to train, val, test set paths
DATASET_PATHS = {
    QualDataset.HUMOD: (ROOT2+'humod/train_HUMOD.json',
                        ROOT2+'humod/val_HUMOD.json',
                        ROOT2+'humod/test_HUMOD.json'),
    QualDataset.USR_TC_REL: (ROOT + 'tc/TopicalChat_train.json',
                             ROOT2 + 'usr_tc/usr_tc_rel_val.json',
                             ROOT2 + 'usr_tc/usr_tc_rel_test.json')
}


class DataLoaderGenerator:

    tr: List[List[object]]  # List[List[BatchEncoding]]
    val: Tuple[List[List[object]], List[float]]  # Tuple[List[List[BatchEncoding]], List[float]]

    load_type: LoaderType

    # If loader does not change from one iteration to the next, then store it here.
    val_loader: Optional[DataLoader] = None
    tr_loader: Optional[DataLoader] = None

    bert_max_len: int
    batchsize: int

    def __init__(self, bert_max_len: int, to_load: QualDataset, load_type: LoaderType, batchsize: int, skip_tr=False, limit=None) -> None:
        """
        Initialize object that can be used to get train or validation loaders.

        <bert_max_len> is the max input size the BERT model can take in tokens.
        <batchsize> is the desired batchsize
        """
        self.load_type = load_type
        self.batchsize = batchsize
        self.bert_max_len = bert_max_len

        train_corpus_path, val_corpus, _ = DATASET_PATHS[to_load]

        if not skip_tr:
            train_corpus = load_corpus(train_corpus_path, load_annot=False)
            if limit is not None and load_type != LoaderType.Rand3750:
                train_corpus.dialogues = train_corpus.dialogues[:limit]
            elif limit is not None:
                # Rand3750 automatically cuts train down to correct size in this case
                assert limit == 3750

        val_corpus = load_corpus(val_corpus, load_annot=True)
        if limit is not None:
            val_corpus.dialogues = val_corpus.dialogues[:limit]

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if not skip_tr and load_type != LoaderType.Rand3750:
            tr_X1 = [tokenizer.encode_plus(" ".join(x.context), x.response, max_length=bert_max_len, pad_to_max_length=True,
                                           return_tensors='pt') for x in train_corpus.dialogues]

        if not skip_tr and load_type == LoaderType.Rand3750:
            tmp = train_corpus.dialogues
            assert (len(tmp) >= 2*3750)
            random.Random(0).shuffle(tmp)

            tr_X1 = [tokenizer.encode_plus(" ".join(x.context), x.response, max_length=bert_max_len, pad_to_max_length=True,
                                           return_tensors='pt') for x in tmp[:3750]]

            tr_X2 = [tokenizer.encode_plus(" ".join(x.context), x.response, max_length=bert_max_len, pad_to_max_length=True,
                                           return_tensors='pt') for x in tmp[3750: 2*3750]]

        if not skip_tr and load_type == LoaderType.Distr0:
            tr_X2 = [tokenizer.encode_plus(" ".join(x.context), x.distractors[0], max_length=bert_max_len,
                                            pad_to_max_length=True, return_tensors='pt') for x in
                     train_corpus.dialogues]

        if not skip_tr and load_type == LoaderType.IDK:
            tr_X2 = [tokenizer.encode_plus(" ".join(x.context), "i don't know.", max_length=bert_max_len,
                                           pad_to_max_length=True, return_tensors='pt') for x in
                     train_corpus.dialogues]

        # Try another non-specific response
        if not skip_tr and load_type == LoaderType.ICS:
            tr_X2 = [tokenizer.encode_plus(" ".join(x.context), "i couldn't say.", max_length=bert_max_len,
                                           pad_to_max_length=True, return_tensors='pt') for x in
                     train_corpus.dialogues]
        if not skip_tr and load_type == LoaderType.OK:
            tr_X2 = [tokenizer.encode_plus(" ".join(x.context), "i'm ok.", max_length=bert_max_len,
                                           pad_to_max_length=True, return_tensors='pt') for x in
                     train_corpus.dialogues]

        # Randomly shuffle to produce negative samples
        #random.shuffle(train_corpus.dialogues)
        #tr_X2 = [tokenizer.encode_plus(" ".join(x.context), x.response, max_length=bert_max_len, pad_to_max_length=True,
        #                               return_tensors='pt') for x in train_corpus.dialogues]
        #tr_X2 = copy.deepcopy(tr_X1)
        #random.shuffle(tr_X2)

        num_distractors = len(val_corpus.distractor_names)

        # Get every response (along with the context)
        val_X = [tokenizer.encode_plus(" ".join(x.context), x.response, max_length=bert_max_len, pad_to_max_length=True,
                                           return_tensors='pt') for x in val_corpus.dialogues]
        for i in range(num_distractors):
            val_X += [tokenizer.encode_plus(" ".join(x.context), x.distractors[i], max_length=bert_max_len,
                                            pad_to_max_length=True, return_tensors='pt') for x in val_corpus.dialogues]

        # Get human score for every response
        val_Y = [x.resp_annot for x in val_corpus.dialogues]
        for i in range(num_distractors):
            val_Y += [x.distr_annots[i] for x in val_corpus.dialogues]

        if not skip_tr:
            if load_type in [LoaderType.Distr0, LoaderType.IDK, LoaderType.Rand3750, LoaderType.ICS, load_type.OK]:
                self.tr = [tr_X1, tr_X2]
            else:
                self.tr = [tr_X1]
        self.val = ([val_X], val_Y)

    def get_val_data_loader(self) -> DataLoader:
        """Return a dataloader for the validation data"""
        if self.val_loader is None:
            val_x, val_y = self.val
            val_x = val_x[0]

            # TODO: Redo this weird kludge
            val_dataset = MyDataset2(val_x, val_y)

            num_val_dialogues = len(val_y)
            val_data_dict = {'tok': torch.zeros((num_val_dialogues, self.bert_max_len)),
                             'att': torch.zeros((num_val_dialogues, self.bert_max_len)),
                             'ty': torch.zeros((num_val_dialogues, self.bert_max_len))}

            for i in range(num_val_dialogues):
                tok, att, ty, _ = val_dataset[i]
                val_data_dict['tok'][i] = tok
                val_data_dict['att'][i] = att
                val_data_dict['ty'][i] = ty

            val_dataset = TensorDataset(val_data_dict['tok'], val_data_dict['att'], val_data_dict['ty'])

            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batchsize, shuffle=False, num_workers=1)
        return self.val_loader

    def get_train_loader(self) -> DataLoader:
        """Return a DataLoader for the training data"""
        def make_shfle_loader() -> DataLoader:
            tr_X1 = self.tr[0]
            tr_X2 = copy.copy(tr_X1)
            random.shuffle(tr_X2)

            train_dataset = MyDataset(tr_X1, tr_X2)
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=1)

        if self.load_type == LoaderType.ShfleTr:
            if self.tr_loader is None:
                self.tr_loader = make_shfle_loader()
            return self.tr_loader
        elif self.load_type in [LoaderType.Distr0, LoaderType.IDK, LoaderType.Rand3750, LoaderType.ICS, LoaderType.OK]:
            if self.tr_loader is None:
                train_dataset = MyDataset(self.tr[0], self.tr[1])
                self.tr_loader = \
                    torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=1)
            return self.tr_loader
        else:
            assert (self.load_type == LoaderType.ShfleTrPerEp)
            return make_shfle_loader()

    def get_val_y(self) -> List[float]:
        return self.val[1]
