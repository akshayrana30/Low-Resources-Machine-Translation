"""
Script to Prepare dataloaders
Ref: a. https://www.tensorflow.org/tutorials/load_data/text
     b. https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""
import io
import os
from datetime import datetime
import tensorflow as tf

from sklearn.model_selection import train_test_split
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


def prepare_training_pairs(path_source,
                           path_target,
                           path_syn_source=None,
                           path_syn_target=None,
                           batch_size=1,
                           valid_ratio=0.2,
                           seed=1234,
                           name="sync"):
    """
    Provide dataloader for translation from aligned training pairs
    """
    tf.random.set_seed(seed)
    # read data line by line with addition of "<start>", "<end>"
    list_source = create_dataset(path_source)
    list_target = create_dataset(path_target)
    print("Sample source", list_source[0])
    print("Sample target", list_target[0])

    # split the dataset into train and valid
    source_train, source_val, target_train, target_val = train_test_split(list_source,
                                                                          list_target,
                                                                          test_size=valid_ratio,
                                                                          random_state=seed)

    if path_syn_source and path_syn_target:
        list_syn_source = create_dataset(path_syn_source)
        list_syn_target = create_dataset(path_syn_target)
        print("Sample syn source", list_syn_source[0])
        print("Sample syn target", list_syn_target[0])
        assert len(list_syn_source) == len(list_syn_target)
        # combine synthetic
        source_train = source_train + list_syn_source
        target_train = target_train + list_syn_target

    # encode text into index of words (only fit tokenizer on training set)
    source_train, source_tokenizer = tokenize(source_train)
    target_train, target_tokenizer = tokenize(target_train)
    source_val = source_tokenizer.texts_to_sequences(source_val)
    target_val = target_tokenizer.texts_to_sequences(target_val)

    size_train = len(source_train)
    size_val = len(source_val)
    print("Size of train set: %s" % size_train)
    print("Size of valid set: %s" % size_val)

    print("Writing the training pairs into files for future evaluation")
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    with open(os.path.join(ROOT_DIR, 'train.lang1_' + timestamp), 'w', encoding="utf-8") as f:
        for src in source_train:
            f.write(convert(source_tokenizer, src[1:-1]) + "\n")

    with open(os.path.join(ROOT_DIR, 'train.lang2+' + timestamp), 'w', encoding="utf-8") as f:
        for tar in target_train:
            f.write(convert(target_tokenizer, tar[1:-1]) + "\n")

    print("Writing the validation pairs into files for future evaluation")
    with open(os.path.join(ROOT_DIR, 'val.lang1_' + timestamp), 'w', encoding="utf-8") as f:
        for src in source_val:
            f.write(convert(source_tokenizer, src[1:-1]) + "\n")

    with open(os.path.join(ROOT_DIR, 'val.lang2_' + timestamp), 'w', encoding="utf-8") as f:
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

    return train_dataset, valid_dataset, source_tokenizer, target_tokenizer, size_train, size_val


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


def prepare_test(path_test, src_tokenizer, batch_size=1):
    # read lines in test files
    list_test = create_dataset(path_test)

    # encode sentences into id
    test_tensor = src_tokenizer.texts_to_sequences(list_test)
    size_test = len(test_tensor)
    test_max_length = max_length(test_tensor)
    print("Size of test: %s" % size_test)
    print("Max length of test set: %s" % test_max_length)

    # Create tf dataset, and optimize input pipeline (shuffle, batch, prefetch)
    test_dataset = tf.data.Dataset.from_generator(lambda: iter(test_tensor), tf.int32).padded_batch(batch_size,
                                                                                                    padded_shapes=[
                                                                                                        None])
    return test_dataset, test_max_length
