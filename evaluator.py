"""
The script that help evaluate the BLEU score of model, provided by TA from Mila IFT6759
(https://github.com/mila-iqia/ift6759/blob/master/projects/project2/tokenizer.py)

The BLEU score here used the implementation from sacreBLEU
(https://github.com/mjpost/sacreBLEU)
"""
import argparse
import subprocess
import tempfile


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).
    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.
    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.
    Returns: None
    """
    # ---------------------------------------------------------------------
    # Include essential module for evaluation
    import os
    import json
    import pickle
    import time
    import tensorflow as tf
    from data.dataloaders import prepare_training_pairs, prepare_test
    from models import Transformer
    from translate import translate_batch
    from definition import ROOT_DIR
    CONFIG = "eval_cfg.json"
    # ---------------------------------------------------------------------
    # Load setting in json file

    with open(os.path.join(ROOT_DIR, CONFIG)) as f:
        para = json.load(f)
    batch_size = para["batch_size"]
    source = para["src"]
    target = para["tar"]
    ckpt_dir = para["ckpt"]

    # ---------------------------------------------------------------------
    # Create test dataloader from input file (tokenized and map to sequence)

    # Todo: The final training and target tokenizer is needed, so as to use the same tokenizer on test data,
    #  because we didn't build a dictionary file.
    f_src = open(source, 'rb')
    f_tar = open(target, 'rb')
    src_tokenizer = pickle.load(f_src)
    tar_tokenizer = pickle.load(f_tar)

    test_dataset, test_max_length = prepare_test(input_file_path,
                                                 src_tokenizer,
                                                 batch_size=batch_size)
    # calculate vocabulary size
    src_vocsize = len(src_tokenizer.word_index) + 1
    tar_vocsize = len(tar_tokenizer.word_index) + 1
    # ---------------------------------------------------------------------
    # Create the instance of model to load checkpoints
    # Todo: Define the model that fit the checkpoints you want to load
    optimizer = tf.keras.optimizers.Adam()
    model = Transformer.Transformer(voc_size_src=src_vocsize,
                                    voc_size_tar=tar_vocsize,
                                    max_pe=10000,
                                    num_encoders=4,
                                    num_decoders=4,
                                    emb_size=512,
                                    num_head=8,
                                    ff_inner=1024)

    # ---------------------------------------------------------------------
    # Load CheckPoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
    # check if loading is successful
    status.assert_existing_objects_matched()

    # ---------------------------------------------------------------------
    # Use Greedy Search to generate prediction and write to pred_file_path
    start = time.time()
    with open(pred_file_path, 'w', encoding='utf-8') as pred_file:
        for (batch, (inp)) in enumerate(test_dataset):
            if batch % 5 == 0:
                print("Evaluating Batch: %s" % batch)
            translation = translate_batch(model, inp, batch_size, tar_tokenizer)
            for sentence in translation:
                pred_file.write(sentence.strip() + '\n')
    end = time.time()
    print("Translation finish in %s s" % (end - start))


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
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')

    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
