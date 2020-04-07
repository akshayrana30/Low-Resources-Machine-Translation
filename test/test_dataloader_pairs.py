"""
script to examine if dataloader for training pairs works fine
"""
import tensorflow as tf
from data.dataloaders import prepare_training_pairs
from models.RNN import RNNEncoder, RNNDecoder


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


source = "../data/pairs/train.lang1"
target = "../data/pairs/train.lang2"

train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, size_train, \
            size_val, source_max_length, target_max_length = prepare_training_pairs(source, target, batch_size=1)

src_vocsize = len(src_tokenizer.word_index) + 1
tar_vocsize = len(tar_tokenizer.word_index) + 1

encoder = RNNEncoder(src_vocsize, 256, 1024, 2)
decoder = RNNDecoder(tar_vocsize, 256, 1024, 2)

for src, tar in train_dataset:
    print("src tensor:", tf.squeeze(src).numpy())
    print("src sentence:", convert(src_tokenizer, tf.squeeze(src).numpy()))
    print("target tensor:", tf.squeeze(tar).numpy())
    print("target sentence:", convert(tar_tokenizer, tf.squeeze(tar).numpy()))
    print("-----------------------------------------------")
    break
