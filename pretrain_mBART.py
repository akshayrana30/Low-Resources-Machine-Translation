"""
Ref: https://arxiv.org/abs/1409.0473
[ICLR 2015] NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
"""
import os
import time
import datetime
import contextlib
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

from data.dataloaders import prepare_mbart_pretrain_pairs
from models import mBART, Transformer

FLAGS = flags.FLAGS

flags.DEFINE_string('corpus', './data/corpus/corpus.multilingual',
                    'path of source language')
flags.DEFINE_string('ckpt', './checkpoints',
                    'path of target language')
flags.DEFINE_integer('seed', 1234,
                     'random seed for reproducible result')
flags.DEFINE_integer('epochs', 100,
                     'number of epochs')
flags.DEFINE_integer('batch_size', 32,
                     'batch size')
flags.DEFINE_integer('num_enc', 8,
                     'number of stacked encoder')
flags.DEFINE_integer('num_dec', 8,
                     'number of stacked decoder')
flags.DEFINE_integer('num_head', 8,
                     'number of head for multi-head attention')
flags.DEFINE_integer('emb_size', 512,
                     'word embedding dimension')
flags.DEFINE_integer('ffnn_dim', 1024,
                     'number of hidden unit for Feed-Forward Neural Networks')
flags.DEFINE_float('valid_ratio', 0.1,
                   'ration of train/valid split')


def main(argv):
    # Creating dataloaders for training and validation
    logging.info("Creating the source dataloader from: %s" % FLAGS.corpus)
    train_dataset, valid_dataset, corpus_tokenizer, size_train, \
    size_val, corpus_max_length = prepare_mbart_pretrain_pairs(path_corpus=FLAGS.corpus,
                                                               batch_size=FLAGS.batch_size,
                                                               valid_ratio=FLAGS.valid_ratio)

    tf.print("corpus max:", corpus_max_length)

    # calculate vocabulary size
    src_vocsize = len(corpus_tokenizer.word_index) + 1
    tar_vocsize = len(corpus_tokenizer.word_index) + 1
    # ----------------------------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Create Transformer Model")
    model = mBART.mBART(voc_size_src=src_vocsize,
                        voc_size_tar=tar_vocsize,
                        max_pe=10000,
                        num_encoders=FLAGS.num_enc,
                        num_decoders=FLAGS.num_dec,
                        emb_size=FLAGS.emb_size,
                        num_head=FLAGS.num_head,
                        ff_inner=FLAGS.ffnn_dim)

    # ----------------------------------------------------------------------------------
    # Choose the Optimizor, Loss Function, and Metrics
    # create custom learning rate schedule
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(1e-3,
                                                                  100000,
                                                                  end_learning_rate=0.0001,
                                                                  power=1.0,
                                                                  cycle=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-6)

    # Todo: figure out why SparceCategorticalCrossentropy
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    def loss_fn(label, pred):
        """
        The criterion above calculate the loss for all words (voc_size), need to mask the loss that
        not appears in label
        """
        mask = tf.math.logical_not(tf.math.equal(label, 0))
        loss = criterion(label, pred)

        # convert the mask from Bool to float
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # ----------------------------------------------------------------------------------
    # train/valid function
    # Todo: need to understand this
    train_step_signature = [
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp):
        # set target
        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]
        # remember the padding
        pad = tf.cast(tf.math.logical_not(tf.math.equal(inp, 0)), tf.int32)
        en = tf.math.equal(inp, 2)
        fr = tf.math.equal(inp, 3)
        # token maskin
        mask = tf.random.uniform(tf.shape(inp))
        mask = tf.math.less(mask, 0.3)
        mask = tf.math.logical_or(tf.math.logical_not(mask), tf.math.logical_or(en, fr))
        mask = tf.cast(mask, tf.int32)
        # [MASK] token index is 1
        inp = tf.math.maximum(inp * mask, 1)
        inp *= pad

        # tf.print("tar inp", tar_inp)
        # tf.print("tar real", tar_real)
        # create mask
        enc_padding_mask = Transformer.create_padding_mask(inp)

        # mask for first attention block in decoder
        look_ahead_mask = Transformer.create_seq_mask(tf.shape(tar_inp)[1])
        dec_target_padding_mask = Transformer.create_padding_mask(tar_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # mask for "enc_dec" multihead attention
        dec_padding_mask = Transformer.create_padding_mask(inp)

        with tf.GradientTape() as tape:
            # feed input into encoder
            predictions = model(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            train_loss = loss_fn(tar_real, predictions)

            # optimize step
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return train_loss

    @tf.function(input_signature=train_step_signature)
    def valid_step(inp):
        # set target
        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]
        # remember the padding
        pad = tf.cast(tf.math.logical_not(tf.math.equal(inp, 0)), tf.int32)
        en = tf.math.equal(inp, 2)
        fr = tf.math.equal(inp, 3)
        # token maskin
        mask = tf.random.uniform(tf.shape(inp))
        mask = tf.math.less(mask, 0.3)
        mask = tf.math.logical_or(tf.math.logical_not(mask), tf.math.logical_or(en, fr))
        mask = tf.cast(mask, tf.int32)
        # [MASK] token index is 1
        inp = tf.math.maximum(inp * mask, 1)
        inp *= pad

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
        val_loss = loss_fn(tar_real, predictions)
        train_accuracy(tar_real, predictions)
        return val_loss

    # ----------------------------------------------------------------------------------
    # Set up Checkpoints, so as to resume training if something interrupt, and save results
    ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt_mBART")
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        ckpt, directory=FLAGS.ckpt, max_to_keep=2
    )
    # restore from latest checkpoint and iteration
    status = ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    # ----------------------------------------------------------------------------------
    # Setup the TensorBoard for better visualization
    logging.info("Setup the TensorBoard...")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/gradient_tape/' + current_time + '/train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # ----------------------------------------------------------------------------------
    # Start Training Process
    EPOCHS = FLAGS.epochs

    for epoch in range(EPOCHS):
        start = time.time()
        total_train_loss = 0.
        total_val_loss = 0.

        # train
        for batch, inp in enumerate(train_dataset):
            train_loss = train_step(inp)
            total_train_loss += train_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, train_loss))
                if batch % 1000 == 0:
                    ckpt.save(file_prefix=ckpt_prefix)

        """
        # save checkpoint
        if (epoch + 1) % 2 == 0:
            ckpt.save(file_prefix=ckpt_prefix)
        """

        # validation
        for batch, inp in enumerate(valid_dataset):
            val_loss = valid_step(inp)
            total_val_loss += val_loss

        # average loss
        total_train_loss /= (size_train / FLAGS.batch_size)
        total_val_loss /= (size_val / FLAGS.batch_size)

        # Write loss to Tensorborad
        with train_summary_writer.as_default():
            tf.summary.scalar('Train loss', total_train_loss, step=epoch)

        with test_summary_writer.as_default():
            tf.summary.scalar('Valid loss', total_val_loss, step=epoch)

        logging.info('Epoch {} Train Loss {:.4f} Valid loss {:.4f} Valid Accuracy {:.4f}'.format(epoch + 1,
                                                                                                 total_train_loss,
                                                                                                 total_val_loss,
                                                                                                 train_accuracy.result()))

        logging.info('Time taken for 1 train_step {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    app.run(main)
