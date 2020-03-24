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

from data.dataloaders import prepare_training_pairs
from models.LSTM_Attention import BiLSTMEncoder, LSTMDecoder

FLAGS = flags.FLAGS

flags.DEFINE_string('source', './data/pairs/train.lang1',
                    'path of source language')
flags.DEFINE_string('target', './data/pairs/train.lang2',
                    'path of target language')
flags.DEFINE_string('ckpt', './checkpoints',
                    'path of target language')
flags.DEFINE_integer('epochs', 100,
                     'number of epochs')
flags.DEFINE_integer('batch_size', 64,
                     'batch size')
flags.DEFINE_integer('enc_units', 64,
                     'internal unit of RNN encoder')
flags.DEFINE_integer('dec_units', 64,
                     'internal unit of RNN decoder')
flags.DEFINE_integer('emb_dim', 128,
                     'word embedding dimension')
flags.DEFINE_float('lr', 1e-3,
                   'learning rate')
flags.DEFINE_float('valid_ratio', 0.2,
                   'ration of train/valid split')


def main(argv):
    # Creating dataloaders for training and validation
    logging.info("Creating the source dataloader from: %s" % FLAGS.source)
    logging.info("Creating the target dataloader from: %s" % FLAGS.target)
    train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, size_train, size_val \
        = prepare_training_pairs(path_source=FLAGS.source,
                                 path_target=FLAGS.target,
                                 batch_size=FLAGS.batch_size,
                                 valid_ratio=FLAGS.valid_ratio)
    # calculate vocabulary size
    src_vocsize = len(src_tokenizer.word_index) + 1
    tar_vocsize = len(tar_tokenizer.word_index) + 1
    # ----------------------------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Create LSTM Model")
    encoder = BiLSTMEncoder(vocab_size=src_vocsize,
                            embedding_dim=FLAGS.emb_dim,
                            enc_units=FLAGS.enc_units,
                            batch_sz=FLAGS.batch_size)
    decoder = LSTMDecoder(vocab_size=tar_vocsize,
                          embedding_dim=FLAGS.emb_dim,
                          dec_units=FLAGS.dec_units,
                          batch_sz=FLAGS.batch_size)

    # ----------------------------------------------------------------------------------
    # Choose the Optimizor, Loss Function, and Metrics
    optimizer = tf.keras.optimizers.Adam()
    # Todo: figure out why SparceCategorticalCrossentropy
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

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

        return tf.reduce_mean(loss)

    # ----------------------------------------------------------------------------------
    # train/valid function
    @tf.function
    def train_step(inp, targ, enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c):
        train_loss = 0
        with tf.GradientTape() as tape:
            # feed input into encoder
            enc_output, enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c = encoder(inp, [enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c])

            # initial states of decoder is the final state of encoder
            dec_hidden = enc_f_hidden
            dec_c = enc_f_c

            # initial input of decoder is the word "<start>", shape: [batch_size, 1]
            dec_input = tf.expand_dims([tar_tokenizer.word_index['<start>']] * FLAGS.batch_size, axis=1)

            # iterate in timestep dim to decode
            for t in range(1, targ.shape[1]):
                pred, dec_hidden, dec_c = decoder(dec_input, [dec_hidden, dec_c], enc_output)
                train_loss += loss_fn(targ[:, t], pred)

                # use last label as next input (teacher forcing)
                dec_input = tf.expand_dims(targ[:, t], axis=1)

            # optimize step
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(train_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
        return train_loss

    @tf.function
    def valid_step(inp, targ, enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c):
        val_loss = 0
        # feed input into encoder
        enc_output, enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c = encoder(inp, [enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c])
        # initial states of decoder is the final state of encoder
        dec_hidden = enc_f_hidden
        dec_c = enc_f_c
        # initial input of decoder is the word "<start>", shape: [batch_size, 1]
        dec_input = tf.expand_dims([tar_tokenizer.word_index['<start>']] * FLAGS.batch_size, axis=-1)

        # iterate in timestep dim to decode
        for t in range(1, targ.shape[1]):
            pred, dec_hidden, dec_c = decoder(dec_input, [dec_hidden, dec_c], enc_output)
            val_loss += loss_fn(targ[:, t], pred)
            # use last label as next input (teacher forcing)
            dec_input = tf.expand_dims(targ[:, t], axis=-1)
        return val_loss

    # ----------------------------------------------------------------------------------
    # Set up Checkpoints, so as to resume training if something interrupt, and save results
    ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt_LSTM")
    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(
        ckpt, directory=FLAGS.ckpt, max_to_keep=3
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
        enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c = encoder.initialize_hidden_state()
        total_train_loss = 0.
        total_val_loss = 0.

        # train
        for (inp, targ) in train_dataset:
            train_loss = train_step(inp, targ, enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c)
            total_train_loss += train_loss

        # save checkpoint
        if (epoch + 1) % 2 == 0:
            ckpt.save(file_prefix=ckpt_prefix)

        # validation
        for (inp, targ) in valid_dataset:
            val_loss = valid_step(inp, targ, enc_f_hidden, enc_f_c, enc_b_hidden, enc_b_c)
            total_val_loss += val_loss

        # average loss
        total_train_loss /= size_train
        total_val_loss /= size_val

        # Write loss to Tensorborad
        with train_summary_writer.as_default():
            tf.summary.scalar('Train loss', total_train_loss, step=epoch)

        with test_summary_writer.as_default():
            tf.summary.scalar('Valid loss', total_val_loss, step=epoch)

        logging.info('Epoch {} Train Loss {:.4f} Valid Loss {:.4f}'.format(epoch + 1,
                                                                           total_train_loss,
                                                                           total_val_loss))

        logging.info('Time taken for 1 train_step {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    app.run(main)
