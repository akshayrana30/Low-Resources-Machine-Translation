"""
The script to create training corpus for mBART
"""
import io
import os

from definition import ROOT_DIR

processed_en = "./data/corpus/processed.en/unaligned.en"
processed_fr = "./data/corpus/tokenized.fr/unaligned.fr"
output = "./data/corpus/corpus.multilingual"


def main():
    en_start = "<En> "
    en_end = " <En>"
    fr_start = "<Fr> "
    fr_end = " <Fr>"
    # load tokenized/remove punctuation english corpus into list and add language id <En>
    en_list = io.open(os.path.join(ROOT_DIR, processed_en), encoding='utf-8').read().strip().split('\n')
    en_list = [en_start + s + en_end for s in en_list]

    # load tokenized french corpus into list and add language id <Fr>
    fr_list = io.open(os.path.join(ROOT_DIR, processed_fr), encoding='utf-8').read().strip().split('\n')
    fr_list = [fr_start + s + fr_end for s in fr_list]
    # combine 2 lists and write a new files corpus.multilingual
    multi_list = en_list + fr_list
    with open(os.path.join(ROOT_DIR, output), 'w', encoding='utf-8') as f:
        for line in multi_list:
            line = line.rstrip()
            f.write(line + "\n")


if __name__ == '__main__':
    main()
