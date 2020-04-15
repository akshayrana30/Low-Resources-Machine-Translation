"""
Transformer Building Blocks
Ref:
[2017] Attention Is All You Need (https://arxiv.org/abs/1706.03762)
The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)
"""
import tensorflow as tf
import numpy as np


class Transformer(tf.keras.Model):
    def __init__(self, voc_size_src, voc_size_tar, max_pe, num_encoders, num_decoders,
                 emb_size, num_head, ff_inner=2048, p_dropout=0.1):
        super(Transformer, self).__init__()
        self.emb_size = emb_size

        self.embedding_src = tf.keras.layers.Embedding(voc_size_src, emb_size, mask_zero=True)
        self.embedding_tar = tf.keras.layers.Embedding(voc_size_tar, emb_size, mask_zero=True)

        self.position_encoding_src = PositionEncoding(max_pe, emb_size)
        self.position_encoding_tar = PositionEncoding(max_pe, emb_size)

        self.encoders = TransformerEncoders(emb_size, num_head, num_encoders, ff_inner, p_dropout)
        self.decoders = TransformerDecoders(emb_size, num_head, num_decoders, ff_inner, p_dropout)

        self.linear = tf.keras.layers.Dense(voc_size_tar)

        # dropout for position encoding like paper said
        self.dropout_enc = tf.keras.layers.Dropout(p_dropout)
        self.dropout_dec = tf.keras.layers.Dropout(p_dropout)

    def call(self, inp_enc, inp_dec, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        # tf.print("inp", inp_enc)
        # tf.print("enc p mask", enc_padding_mask)
        # embedding layer
        enc_output = self.embedding_src(inp_enc)
        # tf.print("emb", enc_output)
        enc_output *= tf.math.sqrt(tf.cast(self.emb_size, tf.float32))
        # tf.print("norl", enc_output)
        enc_output = self.dropout_enc(enc_output, training=training)

        dec_output = self.embedding_tar(inp_dec)
        dec_output *= tf.math.sqrt(tf.cast(self.emb_size, tf.float32))
        dec_output = self.dropout_dec(dec_output, training=training)

        # position encoding
        enc_output = self.position_encoding_src(enc_output)
        # tf.print("enc pos", enc_output)
        dec_output = self.position_encoding_tar(dec_output)

        # encoders
        enc_output = self.encoders(enc_output, training, enc_padding_mask)
        # tf.print("final enc", enc_output[:, 1:-2])
        # decoders
        dec_output = self.decoders(dec_output, training, enc_output, enc_output, look_ahead_mask, dec_padding_mask)

        # linear layer
        output = self.linear(dec_output)
        return output


class TransformerEncoders(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_head, num_encoders=6, ff_inner=2048, p_dropout=0.1):
        super(TransformerEncoders, self).__init__()
        self.num_encoders = num_encoders
        self.encoders = [EncoderUnit(emb_size, num_head, ff_inner, p_dropout) for _ in range(num_encoders)]

    def call(self, x, training, enc_padding_mask):
        for i in range(self.num_encoders):
            x = self.encoders[i](x, training, enc_padding_mask)
            # tf.print("enc", x)
        return x


class TransformerDecoders(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_head, num_decoders=6, ff_inner=1024, p_dropout=0.1):
        super(TransformerDecoders, self).__init__()
        self.num_decoders = num_decoders
        self.decoders = [DecoderUnit(emb_size, num_head, ff_inner, p_dropout) for _ in
                         range(num_decoders)]

    def call(self, x, training, enc_output_k, enc_output_v, look_ahead_mask, dec_padding_mask):
        for i in range(self.num_decoders):
            x = self.decoders[i](x, training, enc_output_k, enc_output_v, look_ahead_mask, dec_padding_mask)
        return x


class EncoderUnit(tf.keras.Model):
    def __init__(self, emb_size, num_head, ff_inner=2048, p_dropout=0.1):
        super(EncoderUnit, self).__init__()

        # multi-head attention layer
        self.attention = MultiHeadAttention(emb_size, num_head)

        # Todo: Layer Norm which axis? -1(emb dimension) or 1(sequence length)
        self.layerNorm_multihead = tf.keras.layers.LayerNormalization()

        # position-wise feedforward neural network layer
        self.ffnn = FeedForwardNN(ff_inner, emb_size)
        self.layerNorm_FFNN = tf.keras.layers.LayerNormalization()

        # regularization by dropout
        self.dropout1 = tf.keras.layers.Dropout(p_dropout)
        self.dropout2 = tf.keras.layers.Dropout(p_dropout)

    def call(self, x, training, enc_padding_mask):
        # x => List of [num_batch, max_length, emb_size] * 3 as Q, K, V
        z = self.attention(x, x, x, enc_padding_mask)
        # tf.print("z", tf.reduce_sum(z))
        z = self.dropout1(z, training=training)
        # Residual Connection and Layer Normalization (cuz x -> [x, x, x])
        z = self.layerNorm_multihead(z + x)
        # tf.print("z", tf.reduce_sum(z))
        # Position-wise Feed-Forward Neural Network
        r = self.ffnn(z)
        r = self.dropout2(r, training=training)
        # Residual Connection and Layer Normalization
        r = self.layerNorm_FFNN(r + z)
        return r


class DecoderUnit(tf.keras.Model):
    def __init__(self, emb_size, num_head, ff_inner=2048, p_dropout=0.1):
        super(DecoderUnit, self).__init__()

        # masked multi-head attention
        self.masked_attention = MultiHeadAttention(emb_size, num_head)
        self.layerNorm_masked = tf.keras.layers.LayerNormalization()

        # encoder-decoder attention
        self.enc_dec_attention = MultiHeadAttention(emb_size, num_head)
        self.layerNorm_enc_dec = tf.keras.layers.LayerNormalization()

        # position-wise feed-forward neural network layer
        self.ffnn = FeedForwardNN(ff_inner, emb_size)
        self.layerNorm_FFNN = tf.keras.layers.LayerNormalization()

        # regularization by dropout
        self.dropout1 = tf.keras.layers.Dropout(p_dropout)
        self.dropout2 = tf.keras.layers.Dropout(p_dropout)
        self.dropout3 = tf.keras.layers.Dropout(p_dropout)

    def call(self, x, training, enc_output_k, enc_output_v, look_ahead_mask, dec_padding_mask):
        output_masked = self.masked_attention(x, x, x, look_ahead_mask)
        output_masked = self.dropout1(output_masked, training=training)
        output_masked = self.layerNorm_masked(output_masked + x)

        output_enc_dec = self.enc_dec_attention(output_masked, enc_output_k, enc_output_v, dec_padding_mask)
        output_enc_dec = self.dropout2(output_enc_dec, training=training)
        output_enc_dec = self.layerNorm_enc_dec(output_enc_dec + output_masked)

        output = self.ffnn(output_enc_dec)
        output = self.dropout3(output, training=training)
        output = self.layerNorm_FFNN(output + output_enc_dec)

        return output


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, emb_size):
        super(PositionEncoding, self).__init__()
        self.position_enc = self._positional_encoding(max_length, emb_size)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + tf.broadcast_to(self.position_enc[:seq_len], tf.shape(x))

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, max_length, emb_size):
        angle_rads = self._get_angles(np.arange(max_length)[:, np.newaxis],
                                      np.arange(emb_size)[np.newaxis, :],
                                      emb_size)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads
        return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_head):
        super(MultiHeadAttention, self).__init__()
        self.WQ = tf.keras.layers.Dense(emb_size)
        self.WK = tf.keras.layers.Dense(emb_size)
        self.WV = tf.keras.layers.Dense(emb_size)

        self.emb_size = emb_size
        self.num_head = num_head

        self.qkv_size = tf.cast(emb_size / num_head, tf.int32)
        self.subattentions = SelfAttention(qkv_size=self.qkv_size)

        # Todo: Make sure what is the output dim of Multi-head Attention
        self.WO = tf.keras.layers.Dense(emb_size)

    def call(self, q, k, v, mask=None):
        # x => [num_batch, max_length, emb_size]
        query = self.WQ(q)
        key = self.WK(k)
        value = self.WV(v)

        # split the Q, K, V for different Head attention
        # [num_batch, max_length, emb_size] => [num_batch, max_length, num_head, qkv_size]
        num_batch = tf.shape(query)[0]
        query = tf.reshape(query, [num_batch, -1, self.num_head, self.qkv_size])
        key = tf.reshape(key, [num_batch, -1, self.num_head, self.qkv_size])
        value = tf.reshape(value, [num_batch, -1, self.num_head, self.qkv_size])

        # convert to [num_batch, num_head, max_length, qkv_size]
        # for individual self attention
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        zs = self.subattentions(query, key, value, mask)
        zs = tf.transpose(zs, perm=[0, 2, 1, 3])
        # convert back to [num_batch, max_length, num_head, qkv_size]
        # and reshape [num_batch, max_length, num_head * qkv_size]
        zs = tf.reshape(zs, [num_batch, -1, self.emb_size])
        zs = self.WO(zs)
        return zs


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, qkv_size):
        super(SelfAttention, self).__init__()
        self.constant_norm = tf.sqrt(tf.cast(qkv_size, tf.float32))

    def call(self, query, key, value, mask=None):
        # x => [num_batch, max_length, emb_size]
        score = tf.linalg.matmul(query, key, transpose_a=False, transpose_b=True)
        # print("score before", score.numpy())
        score = score / self.constant_norm

        # masking before softmax
        if mask is not None:
            # print("mask", mask.numpy())
            score += (mask * -1e9)
        # print("score after", score.numpy())
        score = tf.nn.softmax(score, axis=-1)
        # print("score final", score.numpy())
        z = tf.linalg.matmul(score, value, transpose_a=False, transpose_b=False)
        return z


class FeedForwardNN(tf.keras.layers.Layer):
    def __init__(self, dim_inner, dim_output):
        super(FeedForwardNN, self).__init__()
        self.linear_inner = tf.keras.layers.Dense(dim_inner, activation='relu')
        self.linear_output = tf.keras.layers.Dense(dim_output)

    def call(self, x):
        #  x => [num_batch, max_length, emb_size]
        x = self.linear_inner(x)
        x = self.linear_output(x)
        return x


def create_padding_mask(seq):
    # seq => [num_batch, max_length]
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Todo: Understand how this broadcasting works
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    return mask


def create_seq_mask(seq_len):
    # just a simple upper triangle matrix
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
