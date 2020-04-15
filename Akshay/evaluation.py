import tensorflow as tf
import subprocess
import time
from Transformers_Google import *
from config import *


def evaluate_batch(inp_tensor, targ_lang_tokenizer, transformer,
                   batch_size_for_val, max_length_targ):
  # Expecting input from the val_dataset which is already tokenised.
  
    encoder_input = tf.convert_to_tensor(inp_tensor)
    decoder_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * batch_size_for_val, axis=1)
    output = decoder_input

    for i in range(max_length_targ):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token for all the batches
        if (predicted_id == targ_lang_tokenizer.word_index['<end>']).numpy().all():
            return output, attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return output, attention_weights


def translate_batch(inp, tar, targ_lang_tokenizer, transformer,
                    batch_size_for_val, max_length_targ):
    output,_ = evaluate_batch(inp, targ_lang_tokenizer, transformer, 
                              batch_size_for_val, max_length_targ)
    pred_sentences = targ_lang_tokenizer.sequences_to_texts(output.numpy())
    pred_sentences = [x.split("<end>")[0].replace("<start>","").strip() for x in pred_sentences]
    gold_sentences = targ_lang_tokenizer.sequences_to_texts(tar.numpy())
    gold_sentences = [x.replace('<start> ', "").replace(' <end>', "").replace('<OOV>', "").strip() for x in gold_sentences]
    return gold_sentences, pred_sentences


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


def get_scores(gold_file_path, pred_file_path, targ_lang_tokenizer, 
               val_dataset, transformer, batch_size_for_val, max_length_targ):
    new_start = time.time()
    print("--Saving files to get Bleu Scores--")
    print("Batch Size for Evaluation", batch_size_for_val)
    with open(gold_file_path, 'w', encoding='utf-8', buffering=1) as gold_file, open(pred_file_path, 'w', encoding='utf-8', buffering=1) as pred_file:
        for (batch, (inp, tar)) in enumerate(val_dataset):
            if batch%5==0:
                print("Evaluating for batch", batch)
            gold_fr,pred_fr = translate_batch(inp, tar, targ_lang_tokenizer, 
                                              transformer, batch_size_for_val, max_length_targ)
            for g_fr,p_fr in zip(gold_fr, pred_fr):
                gold_file.write(g_fr.strip() + '\n')
                pred_file.write(p_fr.strip() + '\n')
    
    
    print('Time taken for Evaluation: {} secs\n'.format(time.time() - new_start))
    print("Files saved:", gold_file_path)
    score = compute_bleu(pred_file_path, gold_file_path, False)
    print("Bleu Score: ", score)
    print("-------------")
        