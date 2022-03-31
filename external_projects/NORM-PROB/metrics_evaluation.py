from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import pandas as pd
import argparse
from roberta_mnli.logic_eval_interface import logic_eval
from scipy.stats import pearsonr, spearmanr

def fluency_score(rated_a, opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(opt.pretrained_model_path)
    model = GPT2LMHeadModel.from_pretrained(opt.pretrained_model_path)
    model.to(device)

    model.eval()
    nb_steps, eval_loss, exp_average_loss = 0, 0, None
    score_list = []
    # k = "the book is on the desk. These impressions show , when alive , they had smooth skin , robust limbs with webbed feet , and a ridge of skin on their undersides." tensor(169.6684, device='cuda:0')
    with torch.no_grad():
        for step, s in enumerate(rated_a):  # actually here is a batch with batchsize=1
            # Put model in training mode.
            if not s:
                print('space sentence')
                score_list.append(1e6)
                continue
            s = enc.encode(s)  # + [50256]  #50256 is the token_id for <|endoftext|>
            batch = torch.tensor([s]).to(device)
            loss = model(batch, lm_labels=batch)  # everage -logp
            # print(loss*len(s))
            eval_loss += loss.item()
            nb_steps += 1

            score_list.append(loss.item())

    cutoff = np.quantile([-t for t in score_list], 0.05)
    modified_rating = np.array([cutoff if -t < cutoff else -t for t in score_list])
    normed_rating = (modified_rating - cutoff) / np.abs(cutoff)
    return normed_rating



def context_score(questions, answers, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(opt.pretrained_model_path)
    model = GPT2LMHeadModel.from_pretrained(opt.pretrained_model_path)
    model.to(device)

    model.eval()

    score_list = []
    with torch.no_grad():
        for step, (question,answer) in enumerate(zip(questions, answers)):  # actually here is a batch with batchsize=1
            # Put model in training mode.
            if not answer:
                print('space sentence')
                score_list.append(-1e6)

                continue
            joint_enc = enc.encode(question+' '+answer)  # + [50256]  #50256 is the token_id for <|endoftext|>
            q = enc.encode(question)
            batch_joint = torch.tensor([joint_enc]).to(device)
            batch_q = torch.tensor([q]).to(device)

            loss_joint = model(batch_joint, lm_labels=batch_joint) # everage -logp
            loss_q =  model(batch_q, lm_labels=batch_q)

            p_joint = -loss_joint * (len(joint_enc) -1)
            p_q = -loss_q * (len(q) -1)

            score = p_joint - (p_q)

            score_list.append(score.item())


    cutoff = np.quantile(score_list, 0.05)
    modified_rating = np.array([cutoff if t < cutoff else t for t in score_list])
    normed_rating = (modified_rating - cutoff) / np.abs(cutoff)
    return normed_rating


def diversity_score(cands, ngram):

    score_list = []

    cands = [line.strip() for line in cands]

    for groupid in range(int(len(cands))):
        vocab = {}
        current_sentences = cands[groupid].split('\n')
        for aid in range(len(current_sentences)):
            s = current_sentences[aid]

            if not s:
                continue
            ws = s.split(' ')
            if len(ws)<=ngram:
                k = ' '.join(ws)
                if not k in vocab:
                    vocab[k]=1
                else:
                    vocab[k] = vocab[k]+1
            else:
                for i in range(len(ws)-ngram+1):
                    k = ' '.join(ws[i:i+ngram])
                    if not k in vocab:
                        vocab[k] = 1
                    else:
                        vocab[k] = vocab[k] + 1
        total_num = sum([v for v in vocab.values()])
        # print(total_num)
        # print(max(vocab.values()))
        entropy = 0
        for v in vocab.values():
            entropy += -(v/total_num)*np.log((v/total_num))

        score_list.append(entropy)

    return np.array(score_list)


def logic_consistency(sentences_pre, sentences):

    score_list = logic_eval(sentences_pre, sentences)

    return np.array(score_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained-model-path', default='gpt2', help='model path of pretrained gpt2 finetuned on dataset')
    parser.add_argument('--metric', default='context', help='context | fluency | diversity | logic_consistency')
    parser.add_argument('--file-path', default='context_data_release.csv', help='input .csv file')
    parser.add_argument('--output-file-path', default='test_output_context.csv', help='output .csv file')
    parser.add_argument('--ngram', default=2,
                        help='ngram to be used, for diversity eval')

    opt = parser.parse_args()

    '''
    file follow the following format, no header, answer is generated

    | id 1 | question 1 | answer 1 | 
    | id 2 | question 2 | answer 2 |
    | id 3 | question 3 | answer 3 |
    ...


    '''

    df = pd.read_csv(opt.file_path, header=None)
    print(df)

    if opt.metric == 'context':

        question = df[1].to_list()
        answer = df[2].to_list()
        score_list = context_score(question, answer, opt)


    elif opt.metric == 'fluency':

        sentences = df[2].to_list()
        score_list = fluency_score(sentences, opt)

    elif opt.metric == 'diversity':

        ''' answer in each line consists of a group of sentences conditioned on context, separated by newline'''
        sentences = df[2]
        score_list = diversity_score(sentences, opt.ngram)

    elif opt.metric == 'logic_consistency':
        history = df[1].to_list()
        sentences = df[2].to_list()
        score_list = logic_consistency(history, sentences)
    else:
        raise ValueError("Score: context | fluency | diversity | logic_consistency")

    # hackery
    if opt.file_path == 'context_data_release.csv':
        tmp = df[3].to_numpy()
        for i in [4,5,6,7,8,9,10,11,12]:
            tmp += df[i].to_numpy()
        tmp = tmp/ 10
    else:  # Assume my file, with averaged scores
        tmp = df[3].to_numpy()

    print(spearmanr(score_list, tmp))
    print(pearsonr(score_list, tmp))
    with open('results.txt', 'a') as outfile:
        print('=========================================', file=outfile)
        print(opt, file=outfile)
        print(spearmanr(score_list, tmp), file=outfile)
        print(pearsonr(score_list, tmp), file=outfile)

    df[3] = pd.Series(score_list)
    df.to_csv(opt.output_file_path)
