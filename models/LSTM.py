"""
Ref: https://www.tensorflow.org/tutorials/text/nmt_with_attention
"""
import tensorflow as tf


class LSTMEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(LSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, stateh, statec = self.lstm(x, initial_state=hidden)
        return output,  stateh, statec

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))


class LSTMDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        # linear layer with vocab size
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, stateh, statec = self.lstm(x, initial_state=hidden)
        # In decoder, we feed input 1 by 1, need to reshape for linear layer for predicting
        output = tf.reshape(output, [-1, output.shape[-1]])
        output = self.dense(output)
        return output, stateh, statec
