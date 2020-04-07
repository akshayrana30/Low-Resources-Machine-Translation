"""
Transformer Building Blocks
Ref:
[2017] Attention Is All You Need (https://arxiv.org/abs/1706.03762)
The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)
"""
import tensorflow as tf
import numpy as np


class Transformer(tf.keras.Model):
    def __init__(self, voc_size_src, voc_size_tar, src_max_length, tar_max_length, num_encoders, num_decoders,
                 emb_size, num_head, ff_inner=2048, p_dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding_src = tf.keras.layers.Embedding(voc_size_src, emb_size)
        self.embedding_tar = tf.keras.layers.Embedding(voc_size_tar, emb_size)

        self.position_encoding_src = PositionEncoding(src_max_length, emb_size)
        self.position_encoding_tar = PositionEncoding(tar_max_length, emb_size)

        self.encoders = TransformerEncoders(emb_size, num_head, num_encoders, ff_inner, p_dropout)
        self.decoders = TransformerDecoders(emb_size, num_head, tar_max_length, num_decoders, ff_inner, p_dropout)

        self.linear = tf.keras.layers.Dense(voc_size_tar)

        # dropout for position encoding like paper said
        self.dropout_enc = tf.keras.layers.Dropout(p_dropout)
        self.dropout_dec = tf.keras.layers.Dropout(p_dropout)

    def call(self, inp_enc, inp_dec):
        # embedding layer
        enc_output = self.embedding_src(inp_enc)
        enc_output = self.dropout_enc(enc_output)

        dec_output = self.embedding_tar(inp_dec)
        dec_output = self.dropout_dec(dec_output)

        tf.print("enc:", tf.shape(enc_output))
        # position encoding
        enc_output = self.position_encoding_src(enc_output)
        dec_output = self.position_encoding_tar(dec_output)

        # encoders
        enc_output = self.encoders(enc_output)
        # decoders
        dec_output = self.decoders(dec_output, enc_output, enc_output)

        # linear layer
        output = self.linear(dec_output)
        return output


class TransformerEncoders(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_head, num_encoders=6, ff_inner=2048, p_dropout=0.1):
        super(TransformerEncoders, self).__init__()
        self.encoders = [EncoderUnit(emb_size, num_head, ff_inner, p_dropout) for _ in range(num_encoders)]

    def call(self, x):
        count = 0
        for encoder in self.encoders:
            tf.print("Encoder", count)
            x = encoder(x)
            count += 1
        return x


class TransformerDecoders(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_head, tar_max_length, num_decoders=6, ff_inner=1024, p_dropout=0.1):
        super(TransformerDecoders, self).__init__()
        self.decoders = [DecoderUnit(emb_size, num_head, tar_max_length, ff_inner, p_dropout) for _ in
                         range(num_decoders)]

    def call(self, x, enc_output_k, enc_output_v):
        count = 0
        for decoder in self.decoders:
            tf.print("Decoder", count)
            x = decoder(x, enc_output_k, enc_output_v)
            count += 1
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

    def call(self, x):
        tf.print(tf.shape(x))
        # x => List of [num_batch, max_length, emb_size] * 3 as Q, K, V
        z = self.attention(x, x, x)
        tf.print("after multi:", tf.shape(z))
        z = self.dropout1(z)
        # Residual Connection and Layer Normalization (cuz x -> [x, x, x])
        z = self.layerNorm_multihead(z + x)
        tf.print("after multi layer norm:", tf.shape(z))
        # Position-wise Feed-Forward Neural Network
        r = self.ffnn(z)
        r = self.dropout2(r)
        tf.print("after FF:", tf.shape(r))
        # Residual Connection and Layer Normalization
        r = self.layerNorm_FFNN(r + z)
        tf.print("after FF layer norm:", tf.shape(r))
        return r


class DecoderUnit(tf.keras.Model):
    def __init__(self, emb_size, num_head, tar_max_length, ff_inner=2048, p_dropout=0.1):
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

        self.tar_max_length = tar_max_length

        # regularization by dropout
        self.dropout1 = tf.keras.layers.Dropout(p_dropout)
        self.dropout2 = tf.keras.layers.Dropout(p_dropout)
        self.dropout3 = tf.keras.layers.Dropout(p_dropout)

    def call(self, x, enc_output_k, enc_output_v):
        tf.print(tf.shape(x))
        # get the max_length of input
        seq_mask = create_seq_mask(self.tar_max_length)

        output_masked = self.masked_attention(x, x, x, seq_mask)
        tf.print("output masked:", tf.shape(output_masked))
        output_masked = self.dropout1(output_masked)
        output_masked = self.layerNorm_masked(output_masked + x)
        tf.print("output masked layer norm:", tf.shape(output_masked))

        output_enc_dec = self.enc_dec_attention(output_masked, enc_output_k, enc_output_v)
        output_enc_dec = self.dropout2(output_enc_dec)
        tf.print("output enc dec:", tf.shape(output_enc_dec))
        output_enc_dec = self.layerNorm_enc_dec(output_enc_dec + output_masked)
        tf.print("output enc dec after layer norm:", tf.shape(output_enc_dec))

        output = self.ffnn(output_enc_dec)
        tf.print("after FF:", tf.shape(output))
        output = self.dropout3(output)
        output = self.layerNorm_FFNN(output + output_enc_dec)
        tf.print("after FF layer norm:", tf.shape(output))

        return output


class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, emb_size):
        super(PositionEncoding, self).__init__()
        self.position_enc = self._positional_encoding(max_length, emb_size)

    def call(self, x):
        return x + tf.broadcast_to(self.position_enc, tf.shape(x))

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
        tf.print("query:", tf.shape(query))
        tf.print("K:", tf.shape(k))
        key = self.WK(k)
        tf.print("key:", tf.shape(key))
        value = self.WV(v)
        tf.print("value:", tf.shape(value))

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
        tf.print("output", tf.shape(zs))
        # convert back to [num_batch, max_length, num_head, qkv_size]
        # and reshape [num_batch, max_length, num_head * qkv_size]
        zs = tf.reshape(zs, [num_batch, -1, self.emb_size])
        tf.print("output", tf.shape(zs))
        zs = self.WO(zs)
        tf.print("final output", tf.shape(zs))
        return zs


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, qkv_size):
        super(SelfAttention, self).__init__()
        self.constant_norm = tf.sqrt(tf.cast(qkv_size, tf.float32))

    def call(self, query, key, value, mask=None):
        # x => [num_batch, max_length, emb_size]
        score = tf.linalg.matmul(query, key, transpose_a=False, transpose_b=True)
        score = score / self.constant_norm

        # masking before softmax
        if mask is not None:
            score += (mask * -1e9)

        score = tf.nn.softmax(score, axis=-1)
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
    ss = tf.linalg.matmul(seq, seq, transpose_a=False, transpose_b=True)
    tf.print(seq)
    tf.print(ss)
    mask = tf.expand_dims(seq, axis=-1)
    tf.print(tf.shape(mask))
    mask = tf.reshape(mask, [-1, tf.shape(seq)[1], -1])
    return mask


def create_seq_mask(seq_len):
    # just a simple upper triangle matrix
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
