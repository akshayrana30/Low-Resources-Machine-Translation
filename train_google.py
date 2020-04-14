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
from models import Transformer,google_transformer

FLAGS = flags.FLAGS

flags.DEFINE_string('source', './data/pairs/train.lang1',
                    'path of source language')
flags.DEFINE_string('target', './data/pairs/train.lang2',
                    'path of target language')
flags.DEFINE_string('ckpt', './ckpt_base_transformer',
                    'path of target language')
flags.DEFINE_integer('seed', 1234,
                     'random seed for reproducible result')
flags.DEFINE_integer('epochs', 100,
                     'number of epochs')
flags.DEFINE_integer('batch_size', 1,
                     'batch size')
flags.DEFINE_integer('num_enc', 6,
                     'number of stacked encoder')
flags.DEFINE_integer('num_dec', 6,
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
    logging.info("Creating the source dataloader from: %s" % FLAGS.source)
    logging.info("Creating the target dataloader from: %s" % FLAGS.target)
    train_dataset, valid_dataset, src_tokenizer, tar_tokenizer, size_train, \
    size_val, source_max_length, target_max_length = prepare_training_pairs(path_source=FLAGS.source,
                                                                            path_target=FLAGS.target,
                                                                            batch_size=FLAGS.batch_size,
                                                                            valid_ratio=FLAGS.valid_ratio)

    tf.print("target max", target_max_length)

    # calculate vocabulary size
    src_vocsize = len(src_tokenizer.word_index) + 1
    tar_vocsize = len(tar_tokenizer.word_index) + 1
    # ----------------------------------------------------------------------------------
    # Creating the instance of the model specified.
    logging.info("Create Transformer Model")
    model = google_transformer.Transformer(num_layers=6,
                                           d_model=512,
                                           num_heads=8,
                                           dff=1024,
                                           input_vocab_size=src_vocsize,
                                           target_vocab_size=tar_vocsize,
                                           pe_input=src_vocsize,
                                           pe_target=tar_vocsize,
                                           rate=0.1)
    # ----------------------------------------------------------------------------------
    # Choose the Optimizor, Loss Function, and Metrics
    # create custom learning rate schedule
    class transformer_lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, emb_size, warmup_steps=4000):
            super(transformer_lr_schedule, self).__init__()
            self.emb_size = tf.cast(emb_size, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            lr_option1 = tf.math.rsqrt(step)
            lr_option2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.emb_size) * tf.math.minimum(lr_option1, lr_option2)

    learning_rate = transformer_lr_schedule(FLAGS.emb_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

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
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, targ):
        tar_inp = targ[:, :-1]
        tar_real = targ[:, 1:]

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
            predictions, att = model(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            train_loss = loss_fn(tar_real, predictions)

            # optimize step
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return train_loss

    @tf.function(input_signature=train_step_signature)
    def valid_step(inp, targ):
        tar_inp = targ[:, :-1]
        tar_real = targ[:, 1:]

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
    ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt_base_transformer")
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
    train_log_dir = './logs/gradient_tape/' + current_time + '/base_transformer_train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/base_transformer_test'
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
        for (inp, targ) in train_dataset:
            train_loss = train_step(inp, targ)
            total_train_loss += train_loss

        # save checkpoint
        if (epoch + 1) % 2 == 0:
            ckpt.save(file_prefix=ckpt_prefix)

        # validation
        for (inp, tar) in valid_dataset:
            val_loss = valid_step(inp, tar)
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
