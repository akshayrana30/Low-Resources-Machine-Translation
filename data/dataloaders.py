"""
Script to Prepare dataloaders
Ref: a. https://www.tensorflow.org/tutorials/load_data/text
     b. https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""
import io
import os
import unicodedata
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

from preprocessing import tokenizer, punctuation_remover
from definition import ROOT_DIR


# Todo: Understand this
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence):
    # sentence = unicode_to_ascii(sentence)
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


def create_dataset(path):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    lines = [preprocess_sentence(s) for s in lines]
    return lines


def max_length(tensor):
    return max(len(t) for t in tensor)


# Todo: Understand this. to see if it support choosing top V voc, and put <unk>
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
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
            f.write(convert(source_tokenizer, src[1:np.where(src == src_end_idx)[0][0]]) + "\n")

    with open(os.path.join(ROOT_DIR, './data/pairs/val.lang2'), 'w', encoding="utf-8") as f:
        for tar in target_val:
            f.write(convert(target_tokenizer, tar[1:np.where(tar == tar_end_idx)[0][0]]) + "\n")

    # Create tf dataset, and optimize input pipeline (shuffle, batch, prefetch)
    train_dataset = tf.data.Dataset.from_tensor_slices((source_train, target_train)).shuffle(size_train)
    valid_dataset = tf.data.Dataset.from_tensor_slices((source_val, target_val)).shuffle(size_val)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size=batch_size)
    valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size=batch_size)

    return train_dataset, valid_dataset, source_tokenizer, target_tokenizer, size_train, \
           size_val, source_max_length, target_max_length


def prepare_corpus():
    """
    Provide dataloader for learning language models or embeddings from unaligned corpus
    """
    # corpus is not preprocessed yet
    pass


def prepare_training_pairs_emb():
    """
    Provide dataloader for translation with embedding mapping
    (might combine with "prepare_training_pairs" in the end
    """
    pass


def prepare_test_samples(path_source, batch_size=1):
    """
    Provide dataloader for testing and evaluation, the preprocessed step should be the same as preprare train and val
    """

    # read data line by line with addition of "<start>", "<end>"
    list_source = create_dataset(path_source)
    print("Size of training pairs: %s" % (len(list_source)))

    # Todo: Add <unk> in the future, now we use all the words that appears in the texts
    # encode text into index of words
    source_tensor, source_tokenizer = tokenize(list_source)

    source_max_length = max_length(source_tensor)

    size_test = len(source_tensor)
    print("Size of test set: %s" % size_test)

    # Create tf dataset, and optimize input pipeline (shuffle, batch, prefetch)
    test_dataset = tf.data.Dataset.from_tensor_slices(source_tensor).batch(batch_size)

    return test_dataset, source_tokenizer, size_test, source_max_length
