import io
import tensorflow as tf

from data.dataloaders import prepare_training_pairs, preprocess_sentence
from models import Transformer

MAX_LENGTH = 20
source = "../data/pairs/train.lang1"
target = "../data/pairs/train.lang2"
test = "../data/pairs/dummy_test_lang1"

train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, size_train, \
size_val, source_max_length, target_max_length = prepare_training_pairs(source, target, batch_size=1)

src_vocsize = len(src_tokenizer.word_index) + 1
tar_vocsize = len(tar_tokenizer.word_index) + 1

# load model from checkpoint
optimizer = tf.keras.optimizers.Adam()
model = Transformer.Transformer(voc_size_src=src_vocsize,
                                voc_size_tar=tar_vocsize,
                                src_max_length=source_max_length,
                                tar_max_length=target_max_length,
                                num_encoders=6,
                                num_decoders=6,
                                emb_size=512,
                                num_head=8,
                                ff_inner=1024)

ckpt_dir = "../checkpoints/"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
print("Restore Ckpt Sucessfully!!")


# Evaluation Functions
def evaluate(inp_sentence):
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = preprocess_sentence(inp_sentence).split(' ')
    inp_sentence = [src_tokenizer.word_index[x] for x in inp_sentence]
    print(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tar_tokenizer.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        # create mask
        enc_padding_mask = Transformer.create_padding_mask(encoder_input)

        # mask for first attention block in decoder
        look_ahead_mask = Transformer.create_seq_mask(tf.shape(output)[1])
        dec_target_padding_mask = Transformer.create_padding_mask(output)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # mask for "enc_dec" multihead attention
        dec_padding_mask = Transformer.create_padding_mask(encoder_input)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = model(encoder_input,
                            output,
                            False,
                            enc_padding_mask,
                            combined_mask,
                            dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tar_tokenizer.word_index['<end>']:
            return tf.squeeze(output, axis=0)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def translate(sentence, plot=None):
    result = evaluate(sentence).numpy()
    print(result)
    predicted_sentence = [tar_tokenizer.index_word[i] for i in result]
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        # plot_attention_weights(attention_weights, sentence, result, plot)
        pass
    return predicted_sentence


# Read test file line
lines = io.open(test, encoding='UTF-8').read().strip().split('\n')
lines = [s for s in lines]

# translate each line and save as files
with open('../prediction.txt', 'w') as f:
    for line in lines:
        t = translate(line)
        output = ' '.join(t[1:-1]).encode('utf-8').decode("cp950") + '\n'
        f.write(output)
