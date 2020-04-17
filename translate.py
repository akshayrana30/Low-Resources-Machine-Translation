import tensorflow as tf
from models import Transformer


def translate_batch(model, inp, batch_size, tar_tokenizer):
    batch_max_length = tf.shape(inp)[1]
    decoder_input = tf.expand_dims([tar_tokenizer.word_index['<start>']] * batch_size, axis=1)
    output = decoder_input

    for i in range(batch_max_length + 50):
        # create mask
        enc_padding_mask = Transformer.create_padding_mask(inp)
        # mask for first attention block in decoder
        look_ahead_mask = Transformer.create_seq_mask(tf.shape(output)[1])
        dec_target_padding_mask = Transformer.create_padding_mask(output)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # mask for "enc_dec" multihead attention
        dec_padding_mask = Transformer.create_padding_mask(inp)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = model(inp,
                            output,
                            False,
                            enc_padding_mask,
                            combined_mask,
                            dec_padding_mask)

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if (predicted_id == tar_tokenizer.word_index['<end>']).numpy().all():
            return output
        output = tf.concat([output, predicted_id], axis=-1)
    pred_sentences = tar_tokenizer.sequences_to_texts(output.numpy())
    pred_sentences = [x.replace("<start>", "").strip() for x in pred_sentences]
    return pred_sentences
