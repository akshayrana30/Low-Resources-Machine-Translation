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

train_dataset, valid_dataset, src_tokenizer, tar_tokenizer = prepare_training_pairs(source, target, batch_size=2)

src_vocsize = len(src_tokenizer.word_index) + 1
tar_vocsize = len(tar_tokenizer.word_index) + 1

encoder = RNNEncoder(src_vocsize, 256, 1024, 2)
decoder = RNNDecoder(tar_vocsize, 256, 1024, 2)
encoder.build(input_shape=[[2, 64], [2, 1024]])
encoder.summary()

for src, tar in train_dataset:
    # print("src tensor:", src.numpy())
    # print("src sentence:", convert(src_tokenizer, tf.squeeze(src).numpy()))
    # print("target tensor:", tar.numpy())
    # print("target sentence:", convert(tar_tokenizer, tf.squeeze(tar).numpy()))
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(src, sample_hidden)
    tf.print("enc hidden", tf.shape(sample_hidden))
    output, states = decoder(tf.random.uniform((2, 1)), sample_hidden)
    print("-----------------------------------------------")
    break
