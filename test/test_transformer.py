import tensorflow as tf

from data.dataloaders import prepare_training_pairs
from models import Transformer

source = "../data/pairs/train.lang1"
target = "../data/pairs/train.lang2"

train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, size_train, \
size_val, source_max_length, target_max_length = prepare_training_pairs(source, target, batch_size=2)

src_vocsize = len(src_tokenizer.word_index) + 1
tar_vocsize = len(tar_tokenizer.word_index) + 1
print("Source Language voc size: %s" % src_vocsize)
print("Target Language voc size: %s" % tar_vocsize)

print("Source Language max length: %s" % source_max_length)
print("Target Language max length: %s" % target_max_length)

model = Transformer.Transformer(voc_size_src=src_vocsize,
                                voc_size_tar=tar_vocsize,
                                src_max_length=source_max_length,
                                tar_max_length=target_max_length,
                                num_encoders=1,
                                num_decoders=1,
                                emb_size=64,
                                num_head=8,
                                ff_inner=1024)

tf.random.set_seed(1234)
for src, tar in train_dataset:
    # create mask
    enc_padding_mask = Transformer.create_padding_mask(src)

    # mask for first attention block in decoder
    look_ahead_mask = Transformer.create_seq_mask(target_max_length)
    dec_target_padding_mask = Transformer.create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    # mask for "enc_dec" multihead attention
    dec_padding_mask = Transformer.create_padding_mask(src)

    tf.print("src input", src)
    tf.print("tar input", tar)
    output = model(src, tar, enc_padding_mask, combined_mask, dec_padding_mask)
    # tf.print("model output", tf.shape(output))
    break
