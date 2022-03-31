from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import json
import random
from nltk.corpus import webtext

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval().cuda()

    #prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    #next_sentence = "The sky is blue due to the shorter wavelength of blue light."


    #loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))

    def perform_nsp(prompt, next_sentence, mask=None):
        encoding = tokenizer.encode_plus(prompt, next_sentence, return_tensors='pt', max_length=512).to('cuda')
        with torch.no_grad():
            bert_out = model.bert(**encoding)
            pooled_output = bert_out[1]
            if mask is not None:
                #pooled_output = pooled_output * mask
                pooled_output = torch.where(mask, pooled_output, torch.zeros_like(pooled_output))
                #print(pooled_output)

            logits = model.cls(pooled_output)
            return logits

    dataset = webtext.sents()
    dataset = [' '.join(sent) for sent in dataset]

    sequential_data = dataset
    distractors = dataset

    random.seed(42)

    acc_full = 0
    acc_masked = 0
    total = 0

    humod_mask = torch.load('histogram/mask_7_humod_idk_l1_bce.pt').cuda()
    humod_mask

    for i in range(len(sequential_data) - 1):
        prompt = sequential_data[i]
        next_sentence = sequential_data[i+1]
        rand_sentence = random.choice(distractors)

        logits = perform_nsp(prompt, next_sentence)
        acc_full += bool(logits[0, 0] > logits[0, 1]) # next sentence was original
        logits = perform_nsp(prompt, next_sentence, mask=humod_mask)
        acc_masked += bool(logits[0, 0] > logits[0, 1]) # next sentence was original

        logits = perform_nsp(prompt, rand_sentence)
        #print(rand_sentence)
        acc_full += bool(logits[0, 0] < logits[0, 1]) # next sentence was random
        logits = perform_nsp(prompt, rand_sentence, mask=humod_mask)
        acc_masked += bool(logits[0, 0] < logits[0, 1]) # next sentence was random

        total += 2

        if i % 100 == 0:
            print(f"({i}/{len(dataset)}) ORI ACC: {acc_full/total}")
            print(f"({i}/{len(dataset)}) Masked ACC: {acc_masked/total}")
    print("==================================")
    print("DONE RUNNING!")
    print(f"ORI ACC: {acc_full/total}")
    print(f"Masked ACC: {acc_masked/total}")
