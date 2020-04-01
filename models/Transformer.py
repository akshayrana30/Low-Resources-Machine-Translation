"""
Transformer Building Blocks
Ref:
[2017] Attention Is All You Need (https://arxiv.org/abs/1706.03762)
The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)
"""
import tensorflow as tf


class TransformerEncoder(tf.keras.Model):
    def __init__(self, emb_size, num_head, ff_inner=2048):
        super(TransformerEncoder, self).__init__()
        # position encoding layer

        # multi-head attention layer
        self.attention = MultiHeadAttention(emb_size, num_head)

        # Todo: Layer Norm which axis? -1(emb dimension) or 1(sequence length)
        self.layerNorm_multihead = tf.keras.layers.LayerNormalization()

        # position-wise feedforward neural network layer
        self.ffnn = FeedForwardNN(ff_inner, emb_size)
        self.layerNorm_FFNN = tf.keras.layers.LayerNormalization()

    def call(self, x):
        tf.print(tf.shape(x))
        # x => List of [num_batch, max_length, emb_size] * 3 as Q, K, V
        z = self.attention(x, x, x)
        tf.print("after multi:", tf.shape(z))
        # Residual Connection and Layer Normalization (cuz x -> [x, x, x])
        z = self.layerNorm_multihead(z + x)
        tf.print("after multi layer norm:", tf.shape(z))
        # Position-wise Feed-Forward Neural Network
        r = self.ffnn(z)
        tf.print("after FF:", tf.shape(r))
        # Residual Connection and Layer Normalization
        r = self.layerNorm_FFNN(r + z)
        tf.print("after FF layer norm:", tf.shape(r))
        return r


class TransformerDecoder(tf.keras.Model):
    def __init__(self):
        super(TransformerDecoder, self).__init__()

    def call(self):
        pass


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

    def call(self, q, k, v):
        # x => [num_batch, max_length, emb_size]
        query = self.WQ(q)
        tf.print("query:", tf.shape(query))
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

        zs = self.subattentions(query, key, value)
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

    def call(self, query, key, value):
        # x => [num_batch, max_length, emb_size]
        score = tf.linalg.matmul(query, key, transpose_a=False, transpose_b=True)
        score = score / self.constant_norm
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
    tf.print(seq)
    mask = tf.expand_dims(seq, axis=-1)
    tf.print(tf.shape(mask))
    mask = tf.reshape(mask, [-1, tf.shape(seq)[1], -1])
    return mask


def create_seq_mask(seq):
    pass

