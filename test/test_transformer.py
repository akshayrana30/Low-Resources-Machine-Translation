import tensorflow as tf
from models import Transformer

num__batch = 8
max_length = 100
emb_size = 512

enc = Transformer.TransformerEncoders(emb_size=512, num_head=8, num_encoders=6, ff_inner=1024)
dec = Transformer.TransformerDecoders(emb_size=512, num_head=8, num_decoders=6, ff_inner=1024)


inp = tf.ones([num__batch, max_length, emb_size])
label = tf.ones([num__batch, max_length, emb_size])
output = enc(inp)
output = dec(label, output, output)
