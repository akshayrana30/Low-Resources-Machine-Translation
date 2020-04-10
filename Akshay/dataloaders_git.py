import io
import tensorflow as tf
import unicodedata
import re


import io
import tensorflow as tf
import unicodedata
import re

def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w, lang):
  w = w.strip()
  if lang=="en":
    # This part is required only for english unaligned samples in word2vec
    w = unicode_to_ascii(w.lower())

    # Adding space with punctuation.
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # Removing everything except(letters)
    w = re.sub(r"[^a-z]+", " ", w)

  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w

def create_dataset(path, num_examples=None):
  # This method is called only on unaligned samples. 
  # Since english is already lowered and punctuations are removed, preprocess it as french samples
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = [[preprocess_sentence(w, lang="fr") for w in l.split('\t')]  for l in lines[:num_examples]]
  return zip(*word_pairs)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False)
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
  return tensor, lang_tokenizer




# # def unicode_to_ascii(s):
# #   return ''.join(c for c in unicodedata.normalize('NFD', s)
# #       if unicodedata.category(c) != 'Mn')

# def preprocess_sentence(w):
# #   w = unicode_to_ascii(w.lower().strip())
# #   w = re.sub(r"([?.!,¿])", r" \1 ", w)
# #   w = re.sub(r'[" "]+', " ", w)
# #   w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
#   w = w.strip()
#   w = '<start> ' + w + ' <end>'
#   return w

# def create_dataset(path, num_examples=None):
#   lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
#   word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
#   return zip(*word_pairs)


# def tokenize(lang):
#   lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False )
#   lang_tokenizer.fit_on_texts(lang)

#   tensor = lang_tokenizer.texts_to_sequences(lang)
#   tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
#                                                          padding='post')
#   return tensor, lang_tokenizer