import io
import os
import time
import tensorflow as tf

from definition import ROOT_DIR
from data.dataloaders import prepare_training_pairs, preprocess_sentence, prepare_test
from models import Transformer, google_transformer

source = "../data/pairs/train.lang1"
target = "../data/pairs/train.lang2"

# indicate the file u want to translate here
test = "../data/pairs/val.lang1"
test_target = "../data/pairs/val.lang2"
# we need the original tokenizer so as to preprocess the test data in the same way
train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, \
size_train, size_val = prepare_training_pairs(source, target, batch_size=1, valid_ratio=0.1)

src_vocsize = len(src_tokenizer.word_index) + 1
print(src_vocsize)
tar_vocsize = len(tar_tokenizer.word_index) + 1

test_dataset = prepare_test(test, src_tokenizer, batch_size=64)


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
                                max_pe=10000,
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

MAX_length = 200


def evaluate_batch(inp_tensor, batch_size):
    # Expecting input from the val_dataset which is already tokenised.
    encoder_input = inp_tensor
    tf.print(tf.shape(encoder_input))
    decoder_input = tf.expand_dims([tar_tokenizer.word_index['<start>']] * batch_size, axis=1)
    tf.print(tf.shape(decoder_input))
    output = decoder_input

    for i in range(MAX_length):
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

        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if (predicted_id == tar_tokenizer.word_index['<end>']).numpy().all():
            return output
        output = tf.concat([output, predicted_id], axis=-1)
    return output


def translate_batch(inp, batch_size):
    output = evaluate_batch(inp, batch_size)
    pred_sentences = tar_tokenizer.sequences_to_texts(output.numpy())
    pred_sentences = [x.split("<end>")[0].replace("<start>", "").strip() for x in pred_sentences]
    return pred_sentences


import subprocess


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.
    Returns: None
    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = out.stdout.split(b'\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))
    return sum(scores) / len(scores)


# Read test file line
lines = io.open(source, encoding='UTF-8').read().strip().split('\n')
lines = [s for s in lines]


def convert(lang, tensor):
    s = ""
    for t in tensor:
        if t != 0:
            s += lang.index_word[t] + " "
    return s


pred_file_path = os.path.join(ROOT_DIR, "prediction")
"""
new_start = time.time()
test_target = ""
with  open(pred_file_path, 'w', encoding='utf-8', buffering=1) as pred_file:
    for (batch, (inp)) in enumerate(test_dataset):
        if batch % 5 == 0:
            print("Evaluating for batch", batch)
        pred_fr = translate_batch(inp, batch_size=64)
        for p_fr in pred_fr:
            pred_file.write(p_fr.strip() + '\n')
"""
# print('Time taken for Evaluation: {} secs\n'.format(time.time() - new_start))
print("Files saved:", pred_file_path)
score = compute_bleu(pred_file_path, test_target, False)
print("Bleu Score: ", score)
print("-------------")
