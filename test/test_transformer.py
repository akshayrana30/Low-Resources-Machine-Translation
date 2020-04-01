import tensorflow as tf
from models import Transformer

num__batch = 8
max_length = 100
emb_size = 512

model = Transformer.TransformerEncoder(emb_size=512, num_head=8, ff_inner=1024)

inp = tf.ones([num__batch, max_length, emb_size])
output = model(inp)
