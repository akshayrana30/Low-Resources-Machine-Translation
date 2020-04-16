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
import sentencepiece as spm
from definition import ROOT_DIR


def preprocess_sentence(sentence, start, end):
    sentence = start + sentence + end
    return sentence


def create_dataset(path, preprocess=True, start="<start> ", end=" <end>"):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    if preprocess:
        # for machine translation
        lines = [preprocess_sentence(s, start, end) for s in lines]
    else:
        # for mBART
        lines = [s for s in lines]
    return lines


def max_length(tensor):
    return max(len(t) for t in tensor)


# Todo: Understand this. to see if it support choosing top V voc, and put <unk>
def tokenize(lang, oov_token=None, lower=True):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=lower, oov_token=oov_token)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    return tensor, lang_tokenizer


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


def prepare_training_pairs(path_source, path_target, path_spm, batch_size=1, valid_ratio=0.2, seed=1234, src="<En>", tar="<Fr>"):
    """
    Provide dataloader for translation from aligned training pairs
    """
    tf.random.set_seed(seed)
    # read data line by line with addition of "<start>", "<end>"
    list_source = create_dataset(path_source, start=src+" ", end=" "+src)
    list_target = create_dataset(path_target, start=tar+" ", end=" "+tar)
    print("Size of training pairs: %s" % (len(list_source)))
    sp = spm.SentencePieceProcessor()
    sp.Load(path_spm)
    # encode sentences into id
    list_source = list(map(sp.EncodeAsIds, list_source))
    list_target = list(map(sp.EncodeAsIds, list_target))
    # split the dataset into train and valid
    source_train, source_val, target_train, target_val = train_test_split(list_source,
                                                                          list_target,
                                                                          test_size=valid_ratio, random_state=seed)

    size_train = len(source_train)
    size_val = len(source_val)
    print("Size of train set: %s" % size_train)
    print("Size of valid set: %s" % size_val)

    print("Writing the validation pairs into files for future evaluation")
    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang1'), 'w', encoding="utf-8") as f:
        for src in source_val:
            f.write(sp.DecodeIds(src[2:-1]) + "\n")

    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang2'), 'w', encoding="utf-8") as f:
        for tar in target_val:
            f.write(sp.DecodeIds(tar[2:-1]) + "\n")

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

    return train_dataset, valid_dataset, size_train, size_val


def prepare_mbart_pretrain_pairs(path_corpus, path_spm, batch_size=1, valid_ratio=0.1, seed=1234):
    """
    Provide dataloader for pretraining mBART unaligned corpus
    """
    # Make sure the corpus is tokenized and english's punctuation is removed
    tf.random.set_seed(seed)
    # Todo: adjust preproces funciton
    list_corpus = create_dataset(path_corpus, preprocess=False)
    corpus_max_length = max_length(list_corpus)

    print("Size of training pairs: %s" % (len(list_corpus)))
    sp = spm.SentencePieceProcessor()
    sp.Load(path_spm)
    corpus_tensor = list(map(sp.EncodeAsIds, list_corpus))
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
            None]).shuffle(buffer_size=10000)

    valid_dataset = tf.data.Dataset.from_generator(lambda: iter(source_val), output_types=tf.int32).padded_batch(
        batch_size,
        padded_shapes=[
            None]).shuffle(buffer_size=10000)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, valid_dataset, size_train, size_val, corpus_max_length


def prepare_mbart_finetune(path_corpus, path_source, path_target, batch_size=1, valid_ratio=0.2, seed=1234):
    """
    Provide dataloader for translation from aligned training pairs
    """
    tf.random.set_seed(seed)

    # build tokenizer from corpus
    list_corpus = create_dataset(path_corpus, preprocess=False)
    corpus_tensor, corpus_tokenizer = tokenize(list_corpus, oov_token='[MASK]')

    # read data line by line with addition of "<En>", "<Fr>"
    list_source = create_dataset(path_source, start="<en> ", end=" <en>")
    list_target = create_dataset(path_target, start="<fr> ", end=" <fr>")
    print("Size of training pairs: %s" % (len(list_source)))

    # split the dataset into train and valid
    source_train, source_val, target_train, target_val = train_test_split(list_source,
                                                                          list_target,
                                                                          test_size=valid_ratio, random_state=seed)

    # encode text into index of words (only fit tokenizer on training set)
    source_train = corpus_tokenizer.texts_to_sequences(source_train)
    target_train = corpus_tokenizer.texts_to_sequences(target_train)
    source_val = corpus_tokenizer.texts_to_sequences(source_val)
    target_val = corpus_tokenizer.texts_to_sequences(target_val)

    size_train = len(source_train)
    size_val = len(source_val)
    print("Size of train set: %s" % size_train)
    print("Size of valid set: %s" % size_val)

    print("Writing the validation pairs into files for future evaluation")
    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang1'), 'w', encoding="utf-8") as f:
        for src in source_val:
            f.write(convert(corpus_tokenizer, src[1:-1]) + "\n")

    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang2'), 'w', encoding="utf-8") as f:
        for tar in target_val:
            f.write(convert(corpus_tokenizer, tar[1:-1]) + "\n")

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

    return train_dataset, valid_dataset, corpus_tokenizer, size_train, size_val


def prepare_test(path_test, source_tokenizer, batch_size=1):
    # read lines in test files
    list_source = create_dataset(path_test)
    source_test = source_tokenizer.texts_to_sequences(list_source)
    size_test = len(source_test)
    print("Size of test: %s" % size_test)

    # Create tf dataset, and optimize input pipeline (shuffle, batch, prefetch)
    test_dataset = tf.data.Dataset.from_generator(lambda: iter(source_test), tf.int32).padded_batch(batch_size,
                                                                                                    padded_shapes=[
                                                                                                        None])
    return test_dataset
