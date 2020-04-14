"""
script to examine if dataloader for training pairs works fine
"""
import os
import tensorflow as tf
from data.dataloaders import prepare_mbart_pretrain_pairs
from definition import ROOT_DIR


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


corpus = "../data/corpus/corpus.multilingual"
train_dataset, valid_dataset, corpus_tokenizer, size_train, \
size_val, corpus_max_length = prepare_mbart_pretrain_pairs(corpus, batch_size=2, valid_ratio=0.1)

corpus_vocsize = len(corpus_tokenizer.word_index) + 1

count = 1
for src in train_dataset:

    src = src[0]
    print("src tensor:", tf.squeeze(src).numpy())
    print("src sentence:", convert(corpus_tokenizer, tf.squeeze(src).numpy()))
    print("-----------------------------------------------")
    count += 1
    if count > 10:
        break
