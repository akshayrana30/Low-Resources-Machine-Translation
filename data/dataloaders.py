"""
Script to Prepare dataloaders
Ref: a. https://www.tensorflow.org/tutorials/load_data/text
     b. https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""
import io
import unicodedata

import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

from preprocessing import tokenizer, punctuation_remover


# Todo: Understand this
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence):
    sentence = unicode_to_ascii(sentence)
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


def prepare_training_pairs(path_source, path_target, batch_size=1, valid_ratio=0.2):
    """
    Provide dataloader for translation from aligned training pairs
    """
    tf.random.set_seed(1234)
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
                                                                          test_size=valid_ratio, random_state=1234)
    size_train = len(source_train)
    size_val = len(source_val)
    print("Size of train set: %s" % size_train)
    print("Size of valid set: %s" % size_val)

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
