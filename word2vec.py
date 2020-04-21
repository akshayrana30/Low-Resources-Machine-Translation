from dataloaders_processed import load_data, dataloader_aligned, dataloader_unaligned
from gensim.models import Word2Vec
import argparse
import numpy as np
import config as cfg


def train_model(emb_dim, data):
    model = Word2Vec(size=emb_dim, window=3, min_count=1)
    model.build_vocab(sentences=[x.split(" ") for x in data])
    total_examples = model.corpus_count
    model.train(sentences=[x.split(" ") for x in data], total_examples=total_examples, epochs=5)
    return model


def get_embedding_matrix(tokenizer, word_vector, vocab_size, emb_size=256):
    embedding_matrix = np.zeros((vocab_size, emb_size), dtype="float32")
    for (batch, (index, word)) in enumerate(tokenizer.index_word.items()):
        if batch + 1 == vocab_size:
            break
        embedding_matrix[index] = word_vector[word] if word in word_vector else np.zeros(emb_size, dtype="float32")
    return embedding_matrix


def save_emb(emb, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(emb, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", type=str, help="language to train for ex: \"en\", \"fr\" ", default="en")
    parser.add_argument("emb_dim", type=int, help="Embedding Dimension", default=256)
    parser.add_argument("vocab_size", type=int, help="Size of vocabulary", default=20000)
    args = parser.parse_args()
    lang = args.lang
    emb_dim = args.emb_dim
    vocab_size = args.vocab_size
    _, _, _, input_tokenizer, _, _, _, _, target_tokenizer, _ = load_data(False, False, False, vocab_size, vocab_size,
                                                                          emb_dim)

    a_en, a_fr = dataloader_aligned()
    ua_en, ua_fr = dataloader_unaligned()

    if lang == "en":
        data = a_en + ua_en
        path = cfg.root_path + cfg.emb_path_en
    else:
        data = a_fr + ua_fr
        path = cfg.root_path + cfg.emb_path_fr

    model = train_model(emb_dim, data)
    word_vectors = model.wv
    emb = get_embedding_matrix(input_tokenizer, word_vectors, vocab_size, emb_dim)
    save_emb(emb, path)
