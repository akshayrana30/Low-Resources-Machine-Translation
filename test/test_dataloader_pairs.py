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
path_spm = "../preprocessing/m.model"

train_dataset, valid_dataset, size_train, size_val = prepare_training_pairs(source,
                                                                            target,
                                                                            path_spm,
                                                                            batch_size=2,
                                                                            valid_ratio=0.1)

sp = spm.SentencePieceProcessor()
sp.Load(path_spm)

src_vocsize = len(sp)
tar_vocsize = len(sp)

count = 0
for src, tar in valid_dataset:
    src = src[0]
    tar = tar[0]
    tar_inp = tar[:-2]
    tar_real = tar[2:]
    print(sp.piece_to_id("<Fr>"))
    print("src tensor:", tf.squeeze(src).numpy())
    print("src sentence:", sp.DecodeIds((tf.squeeze(src).numpy().tolist())))
    print("target tensor:", tf.squeeze(tar).numpy())
    print("target sentence:", sp.DecodeIds((tf.squeeze(src).numpy().tolist())))
    print("tar inp sentence:", sp.DecodeIds((tf.squeeze(tar_inp).numpy().tolist())))
    print("tar real sentence:", sp.DecodeIds((tf.squeeze(tar_real).numpy().tolist())))
    print("-----------------------------------------------")
    count+=1
    if count > 20:
        break
