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
    ##### MODIFY BELOW #####
    # Warp the test_evaluation.py as a function in here
    import pickle
    import tensorflow as tf
    from config import d_model
    from Transformers_Google import Transformer, CustomSchedule
    from evaluation import translate_batch
    from dataloaders_processed import load_test_generator

    root_path = ""
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    transformer = Transformer(4, 256, 8, 1024, 20000, 20000,
                              20000, 20000, 0.1, None, None)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, root_path + "model_weights/", max_to_keep=3)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

    input_tokenizer = pickle.load(open(root_path+"tokenizers/input_tokenizer.pkl", "rb" ))
    target_tokenizer = pickle.load(open(root_path+"tokenizers/target_tokenizer.pkl", "rb" ))
    print("Tokenizer loaded")

    batch_size = 256
    test_dataset = load_test_generator(input_file_path,
                                       input_tokenizer, batch_size)
    print("Test generator prepared")

    with open(pred_file_path, 'w', encoding='utf-8', buffering=1) as pred_file:
        for batch, inp in enumerate(test_dataset):
            if batch % 2 == 0:
                print("Evaluating for batch", batch)
            preds = translate_batch(inp, target_tokenizer,
                                    transformer, max_length_targ=120)
            for p_fr in preds:
                pred_file.write(p_fr.strip() + '\n')

    ##### MODIFY ABOVE #####


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
    # print(out)
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
