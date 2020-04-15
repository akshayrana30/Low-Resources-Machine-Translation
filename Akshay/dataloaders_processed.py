    

import io
import re
import unicodedata
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from config import *



def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w, lang, aligned=True, add_special_tag=True):
  w = w.strip()
  
  if lang=="en" and not aligned:
    # This part is required only for english unaligned samples in word2vec
    w = unicode_to_ascii(w.lower())
    
    # Removing everything except(letters)
    w = re.sub(r"[^a-z]+", " ", w)

  if lang=="fr" and not aligned:
    # Adding space with punctuation for easy split.
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

  w = w.strip()

  # currently keeping these tags for both type of data
  if add_special_tag:
    w = '<start> ' + w + ' <end>'
  return w


# def tokenize(lang):
#   lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False)
#   lang_tokenizer.fit_on_texts(lang)
#   tensor = lang_tokenizer.texts_to_sequences(lang)
#   tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
#   return tensor, lang_tokenizer


def load_lang(lang_path):
  return io.open(root_path+lang_path, encoding='UTF-8').read().strip().split('\n')


def dataloader_unaligned():
  unaligned_en = load_lang(unaligned_en_path)
  unaligned_fr = load_lang(unaligned_fr_path)
  unaligned_en = [preprocess_sentence(x, "en", aligned=False) for x in unaligned_en]
  unaligned_fr = [preprocess_sentence(x, "fr", aligned=False) for x in unaligned_fr]
  return unaligned_en, unaligned_fr


def dataloader_aligned(add_special_tag=True):
  aligned_en = load_lang(aligned_en_path)
  aligned_fr = load_lang(aligned_fr_path)
  aligned_en = [preprocess_sentence(x, "en", aligned=True, add_special_tag=add_special_tag) for x in aligned_en]
  aligned_fr = [preprocess_sentence(x, "fr", aligned=True, add_special_tag=add_special_tag) for x in aligned_fr]
  return aligned_en, aligned_fr


def dataloader_aligned_synthetic():
  try:
    aligned_en = load_lang(aligned_en_synth_path)
    aligned_fr = load_lang(aligned_fr_synth_path)
  except Exception as e:
    print(e)
    print("No synthetic data present.")

  aligned_en = [preprocess_sentence(x, "en", aligned=True) for x in aligned_en]
  aligned_fr = [preprocess_sentence(x, "fr", aligned=True) for x in aligned_fr]
  return aligned_en, aligned_fr


def tokenize(aligned_lang, unaligned_lang, num_words):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ',
                                                        lower=False, 
                                                        num_words=num_words)
  lang_tokenizer.fit_on_texts(aligned_lang+unaligned_lang)
  tensor = lang_tokenizer.texts_to_sequences(aligned_lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer


def train_val_split(input_lang, output_lang, seed=1234):
  return train_test_split(input_lang, output_lang, test_size=0.2, random_state=seed)


def load_embeddings(emb_path):
  try:
    with open(root_path + emb_path, "rb") as f:
        emb = pickle.load(f)
    return emb
  except:
    print("Embeddings does not exist: ", emb_path)


def load_data(reverse_translate=False, add_synthetic_data=False, 
              load_emb=False, inp_vocab_size=20000,
              tar_vocab_size=20000, emb_size = 128):

  print("-- Loading Datafiles --")
  unaligned_en, unaligned_fr = dataloader_unaligned()
  aligned_en, aligned_fr = dataloader_aligned()

  input_aligned_train, input_aligned_val, \
  output_aligned_train, output_aligned_val = train_val_split(aligned_en, aligned_fr)

  print("-- Creating Vocabulary and Tokenizing --")
  input_train, input_tokenizer = tokenize(input_aligned_train, unaligned_en, inp_vocab_size)
  output_train, output_tokenizer = tokenize(output_aligned_train, unaligned_fr, tar_vocab_size)

  if add_synthetic_data:
    print("-- Adding Synthetic Data --")
    synth_en, synth_fr = dataloader_aligned_synthetic()
    synth_en = input_tokenizer.texts_to_sequences(synth_en)
    synth_fr = output_tokenizer.texts_to_sequences(synth_fr)
    synth_en = pad_sequences(synth_en, padding='post', maxlen=input_train.shape[1])
    synth_fr = pad_sequences(synth_fr, padding='post', maxlen=output_train.shape[1])
    input_train = np.concatenate((input_train, synth_en, input_train), axis=0)
    output_train = np.concatenate((output_train, synth_fr, output_train), axis=0)
  
  if load_emb:
    print("-- Loading embedding --")
    e_emb = load_embeddings("emb_en_"+str(emb_size)+"_20k.pkl")
    d_emb = load_embeddings("emb_fr_"+str(emb_size)+"_20k.pkl")
  else:
    print("-- Skipping embeddings --")
    e_emb, d_emb = None, None

  if reverse_translate:
      input_train, \
      input_tokenizer, \
      input_aligned_val, \
      e_emb, \
      output_train, \
      output_tokenizer, \
      output_aligned_val, \
      d_emb  = output_train, \
              output_tokenizer, \
              output_aligned_val, \
              d_emb, \
              input_train, \
              input_tokenizer, \
              input_aligned_val, \
              e_emb

  print("-- Tokenizing Validation set --")
  input_val = input_tokenizer.texts_to_sequences(input_aligned_val)
  input_val = pad_sequences(input_val, padding='post', maxlen=input_train.shape[1])

  output_val = output_tokenizer.texts_to_sequences(output_aligned_val)
  output_val = pad_sequences(output_val, padding='post', maxlen=output_train.shape[1])

  return input_train, input_val, input_aligned_val, input_tokenizer, e_emb, \
        output_train, output_val, output_aligned_val, output_tokenizer, d_emb

if __name__ == "__main__":
  pass
  # inp_vocab_size=20000
  # tar_vocab_size=20000
  # reverse_translate=False
  # add_synthetic_data=True
  # load_emb = True
  # emb_size = 256

  # input_train, input_val, \
  # input_text_val, \
  # input_tokenizer, e_emb, \
  # target_train, target_val, \
  # target_text_val, \
  # target_tokenizer, d_emb = load_data(reverse_translate, 
  #                                       add_synthetic_data,
  #                                       load_emb,
  #                                       inp_vocab_size, 
  #                                       tar_vocab_size,
  #                                       emb_size)

