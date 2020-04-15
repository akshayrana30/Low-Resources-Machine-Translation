"""
script to examine if dataloader for training pairs works fine
"""
import os
import tensorflow as tf
from data.dataloaders import prepare_mbart_pretrain_pairs
from definition import ROOT_DIR
import sentencepiece as spm


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != -1:
            s += lang.IdToPiece(int(t)) + " "
    return s


corpus = "../data/corpus/corpus.multilingual"
path_spm = "../preprocessing/m.model"

train_dataset, valid_dataset, size_train, size_val \
    , corpus_max_length = prepare_mbart_pretrain_pairs(corpus, path_spm, batch_size=2, valid_ratio=0.1)

sp = spm.SentencePieceProcessor()
sp.Load(path_spm)
corpus_vocsize = len(sp)

count = 1
for src in train_dataset:
    src = src[0]
    tar_inp = src[:-1]
    tar_real = src[1:]
    print("src sentence:", sp.DecodeIds((tf.squeeze(src).numpy().tolist())))
    # remember the padding
    pad = tf.cast(tf.math.logical_not(tf.math.equal(src, -1)), tf.int32)
    en = tf.math.equal(src, sp.piece_to_id('<En>'))
    fr = tf.math.equal(src, sp.piece_to_id('<Fr>'))
    # token maskin
    mask = tf.random.uniform(tf.shape(src))
    mask = tf.math.less(mask, 0.20)
    mask = tf.math.logical_or(tf.math.logical_not(mask), tf.math.logical_or(en, fr))
    mask = tf.cast(mask, tf.int32)
    # [MASK] token index is 1
    src = tf.math.maximum(src * mask, 1)
    src *= pad
    # restore padding
    # print("src tensor:", tf.squeeze(src).numpy())
    print("enc sentence:", sp.DecodeIds((tf.squeeze(src).numpy().tolist())))
    # print("tar inp tensor:", tf.squeeze(tar_inp).numpy())
    print("tar inp sentence:", sp.DecodeIds((tf.squeeze(tar_inp).numpy().tolist())))
    # print("tar real tensor:", tf.squeeze(tar_real).numpy())
    print("tar real sentence:", sp.DecodeIds((tf.squeeze(tar_real).numpy().tolist())))
    count += 1
    if count > 10:
        break
