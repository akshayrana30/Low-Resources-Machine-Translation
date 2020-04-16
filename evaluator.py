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
    # load input file => create test dataloader => (spm encode)
    from data.dataloaders import prepare_test
    BATCH_SIZE = 128
    path_spm = "./preprocessing/m.model"
    test_dataset, voc_size, test_max_length = prepare_test(input_file_path, path_spm, batch_size=BATCH_SIZE)
    # create model
    from models import mBART
    import tensorflow as tf
    src_vocsize = voc_size
    tar_vocsize = voc_size
    optimizer = tf.keras.optimizers.Adam()
    # All the model we use is follow this setting
    model = mBART.mBART(voc_size_src=src_vocsize,
                        voc_size_tar=tar_vocsize,
                        max_pe=10000,
                        num_encoders=4,
                        num_decoders=4,
                        emb_size=512,
                        num_head=8,
                        ff_inner=1024)

    # Load CheckPoint
    ckpt_dir = "../ckpt_BT/"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
    status.assert_existing_objects_matched()

    # Greedy Search / Beam Search and write to pred_file_path
    import time
    from translate import translate_batch
    start = time.time()
    with open(pred_file_path, 'w', encoding='utf-8') as pred_file:
        for (batch, (inp)) in enumerate(test_dataset):
            print("Evaluating Batch: %s" % batch)
            translation = translate_batch(inp, batch_size=BATCH_SIZE)
            for sentence in translation:
                pred_file.write(sentence.strip() + '\n')
    end = time.time()
    print("Translation finish in %s s"%(end-start))


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
