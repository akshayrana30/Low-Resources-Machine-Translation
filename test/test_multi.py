import tensorflow as tf

tf.random.set_seed(1234)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention1(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention1, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))
        self.wk = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))
        self.wv = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))

        self.dense = tf.keras.layers.Dense(d_model,
                                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                                 seed=1234))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    # ----------------------------------------------------------------------------------------------------------


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_head):
        super(MultiHeadAttention, self).__init__()
        self.WQ = tf.keras.layers.Dense(emb_size,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))
        self.WK = tf.keras.layers.Dense(emb_size,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))
        self.WV = tf.keras.layers.Dense(emb_size,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))

        self.emb_size = emb_size
        self.num_head = num_head

        self.qkv_size = tf.cast(emb_size / num_head, tf.int32)
        self.subattentions = SelfAttention(qkv_size=self.qkv_size)

        # Todo: Make sure what is the output dim of Multi-head Attention
        self.WO = tf.keras.layers.Dense(emb_size,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                              seed=1234))

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


def main():
    emb_size = 16
    max_length = 3
    batch_size = 1
    num_head = 8
    x = tf.random.uniform([batch_size, max_length, emb_size])
    tf.print()
    mha1 = MultiHeadAttention1(emb_size, num_head)
    mha2 = MultiHeadAttention(emb_size, num_head)

    a1, att = mha1(x, x, x)
    a2 = mha2(x, x, x)
    print(a1)
    print(a2)
    # tf.print("mha1", mha1(x, x, x))
    # tf.print("mha2", mha2(x, x, x))


if __name__ == '__main__':
    main()
