## Flags passed : 
## python tokenizer.py --input ../data/corpus/unaligned.en --output ../data/processed/ --lang en --keep-empty-lines
## Question: Do we experiment with the flags? I guess it could make a difference. 
## This one is just to get something fn. The pipeline is set already so we just gotta re-run it
## tokenizer.py --input ../data/corpus/unaligned.en --output ../data/processed/ --lang en --keep-empty-lines

# Tasks:
# en/fr monolingual/pair vocabs. 
# Visualizations on:
# WC (token), Sentence Count, Vocab Size (OOV). The vocabs are word-based atm.
# Visual: Histograms, min/minax/mean/std/median for each, oov

# Data Analysis:
# We have 4 data sources, unaligned and aligned for both the languages.
# Word count/vocab size for all 4
# Sentence length for all 4

# Unaligned: tokenize -> punct remove
# Parallel: punct remove

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bow = []
sentence_length = []
with open('../processed/unaligned.en', "r") as text_file:
    for line in text_file:
        stripped_line = line.strip()
        bow.extend(stripped_line.split())
        sentence_length.append(len(stripped_line.split()))
print("Word Count Total: ", sum(sentence_length))     
vocab = pd.DataFrame(list(zip(bow, bow, [1] * len(bow))), columns =['Word', 'Token','Count']).groupby('Word',sort=False)['Token'].count().to_dict()
print("Vocab Size by Words: ", len(vocab))