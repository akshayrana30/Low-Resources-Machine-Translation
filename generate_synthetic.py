import argparse
import logging
import sentencepiece as spm


def generate_predictions(ckpt, path_spm, input_file_path: str, pred_file_path: str, reverse=False):
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
    path_spm = path_spm
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
    ckpt_dir = ckpt
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
    status.assert_existing_objects_matched()

    sp = spm.SentencePieceProcessor()
    sp.Load(path_spm)

    # Greedy Search / Beam Search and write to pred_file_path
    import time
    from translate import translate_batch
    start = time.time()
    with open(pred_file_path, 'w', encoding='utf-8') as pred_file:
        for (batch, (inp)) in enumerate(test_dataset):
            print("Evaluating Batch: %s" % batch)
            translation = translate_batch(model, inp, BATCH_SIZE, sp, reverse=reverse)
            for sentence in translation:
                pred_file.write(sentence.strip() + '\n')
    end = time.time()
    print("Translation finish in %s s" % (end - start))


def main():
    parser = argparse.ArgumentParser(
        'script to create synthetic data for backtranslation.')
    parser.add_argument('--input', help='file to be translated')
    parser.add_argument('--output', help='path to outputs - will store files here')
    parser.add_argument('--ckpt', help='file to be translated')
    parser.add_argument('--spm', help='path to outputs - will store files here')
    parser.add_argument('--reverse', help='FR to EN?')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Todo: Remember to modified the path of checkpoints in evaluator.py
    generate_predictions(args.ckpt, args.spm, args.input, args.output, args.reverse)


if __name__ == '__main__':
    main()
