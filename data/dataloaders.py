"""
Script to Prepare dataloaders
Ref: a. https://www.tensorflow.org/tutorials/load_data/text
     b. https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""
import io
import os
from random import shuffle
import unicodedata
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

from preprocessing import tokenizer, punctuation_remover
from definition import ROOT_DIR


def preprocess_sentence(sentence):
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


def create_dataset(path, preprocess=True):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    if preprocess:
        # for machine translation
        lines = [preprocess_sentence(s) for s in lines]
    else:
        # for mBART
        lines = [s for s in lines]
    return lines


def max_length(tensor):
    return max(len(t) for t in tensor)


# Todo: Understand this. to see if it support choosing top V voc, and put <unk>
def tokenize(lang, oov_token=None):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=oov_token)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    return tensor, lang_tokenizer


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


def prepare_training_pairs(path_source, path_target, batch_size=1, valid_ratio=0.2, seed=1234):
    """
    Provide dataloader for translation from aligned training pairs
    """
    tf.random.set_seed(seed)
    # read data line by line with addition of "<start>", "<end>"
    list_source = create_dataset(path_source)
    list_target = create_dataset(path_target)
    print("Size of training pairs: %s" % (len(list_source)))

    # Todo: Add <unk> in the future, now we use all the words that appears in the texts
    # encode text into index of words
    source_tensor, source_tokenizer = tokenize(list_source)
    target_tensor, target_tokenizer = tokenize(list_target)

    source_max_length = max_length(source_tensor)
    target_max_length = max_length(target_tensor)

    # split the dataset into train and valid
    source_train, source_val, target_train, target_val = train_test_split(source_tensor,
                                                                          target_tensor,
                                                                          test_size=valid_ratio, random_state=seed)

    size_train = len(source_train)
    size_val = len(source_val)
    print("Size of train set: %s" % size_train)
    print("Size of valid set: %s" % size_val)

    # writing the validation pairs into files for future evaluation
    src_end_idx = source_tokenizer.word_index['<end>']
    tar_end_idx = target_tokenizer.word_index['<end>']

    print("Writing the validation pairs into files for future evaluation")
    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang1'), 'w', encoding="utf-8") as f:
        for src in source_val:
            f.write(convert(source_tokenizer, src[1:-1]) + "\n")

    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang2'), 'w', encoding="utf-8") as f:
        for tar in target_val:
            f.write(convert(target_tokenizer, tar[1:-1]) + "\n")

    # Create tf dataset, and optimize input pipeline (shuffle, batch, prefetch)
    train_src = tf.data.Dataset.from_generator(lambda: iter(source_train), tf.int32).padded_batch(batch_size,
                                                                                                  padded_shapes=[None])
    train_target = tf.data.Dataset.from_generator(lambda: iter(target_train), tf.int32).padded_batch(batch_size,
                                                                                                     padded_shapes=[
                                                                                                         None])
    train_dataset = tf.data.Dataset.zip((train_src, train_target)).shuffle(size_train)

    val_src = tf.data.Dataset.from_generator(lambda: iter(source_val), tf.int32).padded_batch(batch_size,
                                                                                              padded_shapes=[None])
    val_target = tf.data.Dataset.from_generator(lambda: iter(target_val), tf.int32).padded_batch(batch_size,
                                                                                                 padded_shapes=[None])
    valid_dataset = tf.data.Dataset.zip((val_src, val_target)).shuffle(size_val)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, valid_dataset, source_tokenizer, target_tokenizer, size_train, \
           size_val, source_max_length, target_max_length


def prepare_mbart_pretrain_pairs(path_corpus, batch_size=1, valid_ratio=0.1, seed=1234):
    """
    Provide dataloader for pretraining mBART unaligned corpus
    """
    # Make sure the corpus is tokenized and english's punctuation is removed
    tf.random.set_seed(seed)
    # Todo: adjust preproces funciton
    list_corpus = create_dataset(path_corpus, preprocess=False)
    corpus_max_length = max_length(list_corpus)

    print("Size of training pairs: %s" % (len(list_corpus)))
    corpus_tensor, corpus_tokenizer = tokenize(list_corpus, oov_token='[MASK]')
    print("[MASK] index:", corpus_tokenizer.word_index['[MASK]'])

    # split the dataset into train and valid
    source_train, source_val, target_train, target_val = train_test_split(corpus_tensor,
                                                                          corpus_tensor,
                                                                          test_size=valid_ratio, random_state=seed)

    size_train = len(source_train)
    size_val = len(source_val)
    print("Size of train set: %s" % size_train)
    print("Size of valid set: %s" % size_val)

    # Create tf dataset, and optimize input pipeline (shuffle, batch, prefetch)
    train_dataset = tf.data.Dataset.from_generator(lambda: iter(source_train), output_types=tf.int32).padded_batch(
        batch_size,
        padded_shapes=[
            None]).shuffle(size_train)

    valid_dataset = tf.data.Dataset.from_generator(lambda: iter(source_val), output_types=tf.int32).padded_batch(
        batch_size,
        padded_shapes=[
            None]).shuffle(size_val)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, valid_dataset, corpus_tokenizer, size_train, size_val, corpus_max_length
