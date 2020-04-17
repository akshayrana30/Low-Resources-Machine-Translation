"""
script to examine if dataloader for training pairs works fine
"""
import tensorflow as tf
from data.dataloaders import prepare_training_pairs
import sentencepiece as spm


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


source = "../data/pairs/train.lang1"
target = "../data/pairs/train.lang2"

train_dataset, valid_dataset, src_tokenizer, \
tar_tokenizer, size_train, size_val = prepare_training_pairs(source,
                                                             target,
                                                             batch_size=2,
                                                             valid_ratio=0.1)

count = 0
for src, tar in valid_dataset:
    src = src[0]
    tar = tar[0]
    tar_inp = tar[:-1]
    tar_real = tar[1:]
    end = tf.cast(tf.math.logical_not(tf.math.equal(tar_inp, tar_tokenizer.word_index['<end>'])), tf.int32)
    tar_inp *= end
    print("src tensor:", tf.squeeze(src).numpy())
    print("src sentence:", convert(src_tokenizer, tf.squeeze(src).numpy()))
    print("target tensor:", tf.squeeze(tar).numpy())
    print("target sentence:", convert(tar_tokenizer, tf.squeeze(tar).numpy()))
    print("tar inp sentence:", convert(tar_tokenizer, tf.squeeze(tar_inp).numpy()))
    print("tar real sentence:", convert(tar_tokenizer, tf.squeeze(tar_real).numpy()))
    print("-----------------------------------------------")
    count += 1
    if count > 20:
        break
