from utils.CorpusRepr import load_corpus
import pandas as pd
import os

if __name__ == '__main__':
    for dir, file in [('humod', 'val_HUMOD.json'),  ('usr_tc', 'usr_tc_rel_val.json'),
                      ('humod', 'test_HUMOD.json'),  ('usr_tc', 'usr_tc_rel_test.json'),
                      ('fed', 'fed_cor.json'), ('fed', 'fed_rel.json')]:
        rows=[]
        data = load_corpus(os.path.join(dir, file), load_annot=True)
        os.makedirs('csv', exist_ok=True)
        #with open(os.path.join('csv', file.replace('.json', '.csv')), 'w') as outfile:
        for d in data.dialogues:
            gold_resp = d.response
            if gold_resp == '':
                gold_resp = '...'
            #print(f',"{d.get_context()}","{gold_resp}",{d.resp_annot}', file=outfile)
            rows.append([d.get_context(), gold_resp, d.resp_annot])
            for distr, annot in zip(d.distractors, d.distr_annots):
                #    print(f',"{d.get_context()}","{distr}",{annot}', file=outfile)
                rows.append([d.get_context(), distr, annot])

        df = pd.DataFrame(rows, columns=['cont', 'resp', 'annot'])
        df.to_csv(os.path.join('csv', file.replace('.json', '.csv')), header=False)

