import tensorflow as tf
from models import Transformer


class mBART(tf.keras.Model):
    def __init__(self, voc_size_src, voc_size_tar, max_pe , num_encoders, num_decoders,
                 emb_size, num_head, ff_inner=2048, p_dropout=0.1):
        super(mBART, self).__init__()
        self.transformer = Transformer.Transformer(voc_size_src, voc_size_tar, max_pe,
                                                   num_encoders, num_decoders, emb_size, num_head, ff_inner, p_dropout)

    def call(self, inp_enc, inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        return self.transformer(inp_enc, inp_dec, training, enc_padding_mask, look_ahead_mask, dec_padding_mask)
