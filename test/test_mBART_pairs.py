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
    # remember the padding
    pad = tf.cast(tf.math.logical_not(tf.math.equal(src, 0)), tf.int32)
    en = tf.math.equal(src, 2)
    fr = tf.math.equal(src, 3)
    # token maskin
    mask = tf.random.uniform(tf.shape(src))
    mask = tf.math.less(mask, 0.3)
    mask = tf.math.logical_or(tf.math.logical_not(mask), tf.math.logical_or(en, fr))
    mask = tf.cast(mask, tf.int32)
    # [MASK] token index is 1
    src = tf.math.maximum(src * mask, 1)
    src *= pad
    # restore padding
    print("src tensor:", tf.squeeze(src).numpy())
    print("src sentence:", convert(corpus_tokenizer, tf.squeeze(src).numpy()))
    print("-----------------------------------------------")
    count += 1
    if count > 10:
        break
