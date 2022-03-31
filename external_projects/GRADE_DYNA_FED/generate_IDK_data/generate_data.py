"""
Simple script, convert datasets from my corpus format to the author's format.
"""

import json
import os


def main():
    tmp_header = None

    for  dir, filename, childname in [("humod", "test_HUMOD.json", "test"),
                                      ("humod", "val_HUMOD.json", "val"),
                                      ("usr_tc", "usr_tc_rel_test.json", "test"),
                                      ("usr_tc", "usr_tc_rel_val.json", "val"),
                                      ("fed", "fed_cor.json", "cor"),
                                      ("fed", "fed_rel.json", "rel"),
                                      ("pang_dd", "pang_dd.json", "test")]:
        if tmp_header is None:
            tmp_header=f"DATA=(IDK_{dir}_{childname}"
        else:
            tmp_header+= f" IDK_{dir}_{childname}"


        with open(os.path.join("generate_IDK_data", "data", dir, filename), 'r') as infile:
            dataset = json.load(infile)

        dial_eval_dataset = {"contexts": [], # List of list of str
                             "responses": [],  # List of str
                             "references": [], # List of "NO REF" repeated
                             "scores": [], # List of float
                             }

        for dial in dataset['dialogues']:
            dial_eval_dataset["contexts"].append(dial['context'])
            dial_eval_dataset['responses'].append(dial['response'])
            dial_eval_dataset['references'].append("NO REF")
            dial_eval_dataset['scores'].append(dial['resp_annot'])

            for distr_resp, distr_annot in zip(dial["distractors"], dial["distr_annots"]):
                dial_eval_dataset["contexts"].append(dial['context'])
                dial_eval_dataset['responses'].append(distr_resp)
                dial_eval_dataset['references'].append("NO REF")
                dial_eval_dataset['scores'].append(distr_annot)

        os.makedirs(os.path.join("data", f"IDK_{dir}_{childname}_data"), exist_ok=True)
        with open(os.path.join("data", f"IDK_{dir}_{childname}_data", f"IDK_{dir}_{childname}.json"), 'w') as outfile:
            json.dump(dial_eval_dataset, outfile, indent=2)

        print(f"""
def load_IDK_{dir}_{childname}_data(basedir):
    with open(os.path.join(basedir, "IDK_{dir}_{childname}.json"), 'r') as infile:
    return json.load(infile)
        """)

    tmp_header += ")"
    print(tmp_header)

if __name__ == '__main__':
    main()
