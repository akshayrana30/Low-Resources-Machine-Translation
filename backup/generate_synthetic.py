from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', './ckpt',
                    'path of source language')
flags.DEFINE_string('src', './data/pairs/train.lang1',
                    'path of source language')
flags.DEFINE_string('tar', './data/pairs/train.lang2',
                    'path of target language')
flags.DEFINE_string('input', None,
                    'path of target language')
flags.DEFINE_string('output', './ckpt_pretrain_mBART',
                    'path of target language')
flags.DEFINE_integer('num_syn', 30000,
                     'number of synthetic data u want')


def generate_predictions(ckpt, path_src, path_tar, input_file_path: str, pred_file_path: str, num_sync):
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
    TOTAL_ITER = int((num_sync / 128))

    # load  tokenizer of train to tokenize test data
    import pickle
    f_src = open(path_src, 'rb')
    f_tar = open(path_tar, 'rb')
    src_tokenizer = pickle.load(f_src)
    tar_tokenizer = pickle.load(f_tar)

    test_dataset, test_max_length = prepare_test(input_file_path, src_tokenizer, batch_size=BATCH_SIZE)
    # create model
    from models import Transformer
    import tensorflow as tf
    src_vocsize = len(src_tokenizer.word_index) + 1
    tar_vocsize = len(tar_tokenizer.word_index) + 1
    # create model instance
    optimizer = tf.keras.optimizers.Adam()
    model = Transformer.Transformer(voc_size_src=src_vocsize,
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

    # Greedy Search / Beam Search and write to pred_file_path
    import time
    from translate import translate_batch
    start = time.time()
    count = 0
    with open(pred_file_path, 'w', encoding='utf-8') as pred_file:
        for (batch, (inp)) in enumerate(test_dataset):
            print("Evaluating Batch: %s" % batch)
            batch_size = tf.shape(inp)[0].numpy()
            translation = translate_batch(model, inp, batch_size, tar_tokenizer)
            for sentence in translation:
                pred_file.write(sentence.strip() + '\n')
                pred_file.flush()
            count += 1
            if count > TOTAL_ITER:
                break
    end = time.time()
    print("Translation finish in %s s" % (end - start))


def main(argv):
    # Todo: Remember to modified the path of checkpoints in evaluator.py
    generate_predictions(FLAGS.ckpt, FLAGS.src, FLAGS.tar, FLAGS.input, FLAGS.output, FLAGS.num_syn)


if __name__ == '__main__':
    app.run(main)
