"""
Ref: https://arxiv.org/abs/1409.0473
[ICLR 2015] NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
"""
import tensorflow as tf


class BiLSTMEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(BiLSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,
                                                                         return_sequences=True,
                                                                         return_state=True,
                                                                         recurrent_initializer='glorot_uniform'),
                                                    merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.BiLSTM(x, initial_state=hidden)
        return output, forward_h, forward_c, backward_h, backward_c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)) for i in range(4)]


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.Va = tf.keras.layers.Dense(1)

    def call(self, prev_dec_hidden, enc_output):
        prev_dec_hidden = tf.expand_dims(prev_dec_hidden, axis=1)
        e = self.Va(tf.nn.tanh(self.W(enc_output) + self.U(prev_dec_hidden)))
        attention_weights = tf.nn.softmax(e, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


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
        # attention Layer
        self.att = Attention(dec_units)

        # linear layer with vocab size
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.att(hidden[0], enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, stateh, statec = self.lstm(x, initial_state=hidden)
        # In decoder, we feed input 1 by 1, need to reshape for linear layer for predicting
        output = tf.reshape(output, [-1, output.shape[-1]])
        output = self.dense(output)
        return output, stateh, statec
