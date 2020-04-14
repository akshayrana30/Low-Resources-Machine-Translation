import io
import os
import tensorflow as tf

from definition import ROOT_DIR
from data.dataloaders import prepare_training_pairs, preprocess_sentence
from models import Transformer

source = "../data/pairs/train.lang1"
target = "../data/pairs/train.lang2"

# indicate the file u want to translate here
test = "../data/pairs/val.lang1"

# we need the original tokenizer so as to preprocess the test data in the same way
train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, size_train, \
size_val, source_max_length, target_max_length = prepare_training_pairs(source, target, batch_size=1, valid_ratio=0.1)

src_vocsize = len(src_tokenizer.word_index) + 1
tar_vocsize = len(tar_tokenizer.word_index) + 1

print("ma length", target_max_length)


# load model from checkpoint (we can directly load the model here if we don't use check points)
class transformer_lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, emb_size, warmup_steps=4000):
        super(transformer_lr_schedule, self).__init__()
        self.emb_size = tf.cast(emb_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        lr_option1 = tf.math.rsqrt(step)
        lr_option2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.emb_size) * tf.math.minimum(lr_option1, lr_option2)


learning_rate = transformer_lr_schedule(512)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
model = Transformer.Transformer(voc_size_src=src_vocsize,
                                voc_size_tar=tar_vocsize,
                                src_max_length=source_max_length,
                                tar_max_length=10000,
                                num_encoders=6,
                                num_decoders=6,
                                emb_size=512,
                                num_head=8,
                                ff_inner=1024)

ckpt_dir = "../ckpt_base_transformer/"
latest = tf.train.latest_checkpoint(ckpt_dir)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
status.assert_existing_objects_matched()


# Evaluation Functions
def evaluate(inp_sentence, max_length):
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = preprocess_sentence(inp_sentence).split(' ')
    inp_sentence = [src_tokenizer.word_index[x] for x in inp_sentence]
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tar_tokenizer.word_index['<start>']]
    output = tf.expand_dims(decoder_input, 0)
    for i in range(max_length):
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
        if predicted_id[0][0] == tar_tokenizer.word_index['<end>']:
            return tf.squeeze(output, axis=0)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def translate(sentence, max_length, plot=None):
    result = evaluate(sentence, max_length).numpy()
    print(result)
    predicted_sentence = [tar_tokenizer.index_word[i] for i in result]
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        # plot_attention_weights(attention_weights, sentence, result, plot)
        pass
    return predicted_sentence


# Read test file line
lines = io.open(source, encoding='UTF-8').read().strip().split('\n')
lines = [s for s in lines]


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s

"""
count = 0
for inp, targ in valid_dataset:
    # print("src tensor:", tf.squeeze(inp).numpy())
    print("src sentence:", convert(src_tokenizer, tf.squeeze(inp).numpy()))
    # print("target tensor:", tf.squeeze(targ).numpy())
    print("target sentence:", convert(tar_tokenizer, tf.squeeze(targ).numpy()))
    tar_inp = targ[:, :-1]
    print("input:", convert(tar_tokenizer, tf.squeeze(tar_inp).numpy()))
    tar_real = targ[:, 1:]

    # inp = tf.ones_like(inp)
    # create mask
    enc_padding_mask = Transformer.create_padding_mask(inp)
    # mask for first attention block in decoder
    look_ahead_mask = Transformer.create_seq_mask(tf.shape(tar_inp)[1])
    dec_target_padding_mask = Transformer.create_padding_mask(tar_inp)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    # mask for "enc_dec" multihead attention
    dec_padding_mask = Transformer.create_padding_mask(inp)

    # feed input into encoder
    predictions = model(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
    predictions = tf.argmax(predictions, axis=-1)
    print("predicted sentence:", convert(tar_tokenizer, tf.squeeze(predictions).numpy()))
    print("-----------------------------------------------")
    count += 1
    if count:
        break
"""

# translate each line and save as files
with open(os.path.join(ROOT_DIR, 'base_transformer_prediction.txt'), 'w', encoding='utf-8') as f:
    count = 0
    for line in lines:
        line = line.rstrip()
        print(line)
        # the paper set MAX_LENGTH = input length + 50 when inference
        max_length = 200
        t = translate(line, max_length=max_length)
        output = ' '.join(t[1:-1]) + '\n'
        f.write(output)
        count += 1
        if count > 1:
            break

