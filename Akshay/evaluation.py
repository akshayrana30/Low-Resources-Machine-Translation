import tensorflow as tf
import subprocess
import time
from Transformers_Google import *
from config import *


def evaluate_batch(inp_tensor, targ_lang_tokenizer, transformer, max_length_targ):
  # Expecting input from the val_dataset which is already tokenised.
  
    encoder_input = tf.convert_to_tensor(inp_tensor)
    decoder_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * encoder_input.shape[0], axis=1)
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


def translate_batch(inp, targ_lang_tokenizer, transformer, max_length_targ):
    output,_ = evaluate_batch(inp, targ_lang_tokenizer, transformer, max_length_targ)
    pred_sentences = targ_lang_tokenizer.sequences_to_texts(output.numpy())
    pred_sentences = [x.split("<end>")[0].replace("<start>","").strip() for x in pred_sentences]
    return pred_sentences


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


def get_scores(gold_file_path, pred_file_path, target_tokenizer, 
               val_dataset, target_text_val, transformer, batch_size_for_val, max_length_targ):
    new_start = time.time()
    print("--Saving files to get Bleu Scores--")
    print("Batch Size for Evaluation", batch_size_for_val)
    index = 0
    with open(gold_file_path, 'w', encoding='utf-8', buffering=1) as gold_file, open(pred_file_path, 'w', encoding='utf-8', buffering=1) as pred_file:
        for (batch, (inp, _)) in enumerate(val_dataset):
            if batch%5==0:
                print("Evaluating for batch", batch)
            preds = translate_batch(inp, target_tokenizer, transformer, batch_size_for_val, max_length_targ)
            target = target_text_val[index:index + batch_size_for_val]
            index += batch_size_for_val
            for g_fr,p_fr in zip(target, preds):
                gold_file.write(g_fr.strip() + '\n')
                pred_file.write(p_fr.strip() + '\n')
    
    print('Time taken for Evaluation: {} secs\n'.format(time.time() - new_start))
    print("Files saved:", gold_file_path)
    score = compute_bleu(pred_file_path, gold_file_path, False)
    print("Bleu Score: ", score)
    print("-------------")


def generate_evaluations(transformer, input_path, output_path, 
                         dataset, inp_lang_tokenizer, targ_lang_tokenizer, max_length_targ):
    with open(input_path, 'w', encoding='utf-8', buffering=1) as i_file, open(output_path, 'w', encoding='utf-8', buffering=1) as o_file:
        for batch, inp in enumerate(dataset):
            if batch%50==0:
                print("Generating for batch", batch)
            predicted = translate_batch(inp, targ_lang_tokenizer, transformer,
                                        dataset._batch_size.numpy(), max_length_targ)
            converted_token_to_text = [x.split("<end>")[0].replace("<start>","").strip() for x in inp_lang_tokenizer.sequences_to_texts(inp.numpy())]
            for g_fr,p_fr in zip(converted_token_to_text, predicted):
                i_file.write(g_fr.strip() + '\n')
                o_file.write(p_fr.strip() + '\n')    