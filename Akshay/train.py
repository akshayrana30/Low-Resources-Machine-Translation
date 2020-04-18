from matplotlib import pyplot as plt
import numpy as np
import time
import tensorflow as tf
from Transformers_Google import *
from evaluation import * 
from dataloaders_processed import *
from config import *


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                                                from_logits=True, reduction='none')


def load_dataset():
    input_train, input_val, \
    input_text_val, \
    input_tokenizer, e_emb, \
    target_train, target_val, \
    target_text_val, \
    target_tokenizer, d_emb = load_data(reverse_translate, 
                                        add_synthetic_data,
                                        load_emb,
                                        inp_vocab_size, 
                                        tar_vocab_size,
                                        emb_size)

    max_length_targ, max_length_inp = max_length(target_train), max_length(input_train)
    BUFFER_SIZE = input_train.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(train_batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))
    val_dataset = val_dataset.batch(val_batch_size, drop_remainder=True)

    return train_dataset, val_dataset, input_tokenizer, target_tokenizer, \
    e_emb, d_emb, max_length_inp, max_length_targ, target_text_val


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
    
    
def val_step(inp, tar, transformer, val_loss, val_accuracy):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp, 
                                 False, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)
        
    val_loss(loss)
    val_accuracy(tar_real, predictions)


def max_length(tensor):
    return max(len(t) for t in tensor)


def train():
    train_dataset, val_dataset, input_tokenizer, \
    target_tokenizer, e_emb, d_emb, \
    max_length_inp, max_length_targ, target_text_val = load_dataset()
    print("-- Tf Dataset created --")

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                                                name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                                                name='val_accuracy')

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                            inp_vocab_size, tar_vocab_size, 
                            pe_input=inp_vocab_size, 
                            pe_target=tar_vocab_size,
                            rate=dropout_rate,
                            e_emb=e_emb, d_emb=d_emb)


    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and load_from_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')
    
    print("-- Training Started --")
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar, transformer, 
                       optimizer, train_loss, train_accuracy)
        
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Train Loss {:.4f} Train Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch+1)%evaluate_val_loss_every==0:
            for (batch, (inp, tar)) in enumerate(val_dataset):
                val_step(inp, tar, transformer, 
                         val_loss, val_accuracy)

            print ('Epoch {} Train Loss {:.4f} Train Accuracy {:.4f} Val Loss {:.4f} Val Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result(),
                                                val_loss.result(),                                                    
                                                val_accuracy.result()))

        else:
            print ('Epoch {} Train Loss {:.4f} Train Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result())) 

        if (epoch+1)%save_every==0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

        if (epoch+1)%evaluate_bleu_every==0:
            gold_file_path = root_path+"temp_gold_epoch_"+str(epoch+1)+".txt"
            pred_file_path = root_path+"temp_pred_epoch_"+str(epoch+1)+".txt"
            get_scores(gold_file_path, pred_file_path, target_tokenizer, 
                        val_dataset, target_text_val, transformer, val_batch_size, max_length_targ)
        
        print('Time taken for Epoch: {} secs\n'.format(time.time() - start))

    return transformer, input_tokenizer, target_tokenizer, max_length_inp, max_length_targ


def generate(transformer, input_tokenizer, target_tokenizer, max_length_inp, max_length_targ):
    if reverse_translate:
        # input is french, and output is english
        output_lang, input_lang = dataloader_unaligned()
        print("-- Preparing data to translate French to English --")
    else:
        # input is english, and output is French
        input_lang, output_lang = dataloader_unaligned()
        print("-- Preparing data to translate English to French  --")

    input_path = generate_input_path
    output_path = generate_output_path

    sorted_lines = [x for x in sorted(input_lang, key = len)]
    n = int(number_of_samples/3)
    filtered_lines = sorted_lines[250000:250000+n] + \
                     sorted_lines[350000:350000+n] + \
                     sorted_lines[400000:400000+n]
    
    import random
    random.shuffle(filtered_lines)
    
    input_lang = input_tokenizer.texts_to_sequences(filtered_lines)
    input_lang = tf.keras.preprocessing.sequence.pad_sequences(input_lang,
                                                               padding='post', 
                                                               maxlen=max_length_inp)
    

    new_dataset = tf.data.Dataset.from_tensor_slices(input_lang).shuffle(val_batch_size)
    new_dataset = new_dataset.batch(val_batch_size, drop_remainder=True)

    print("-- Dataset Generated for translation --")
    generate_evaluations(transformer, input_path, 
                        output_path, new_dataset, 
                        input_tokenizer, target_tokenizer, 
                        max_length_targ)




if __name__ == "__main__":
        transformer, input_tokenizer, \
        target_tokenizer, max_length_inp, max_length_targ = train()
        if generate_samples:
            generate(transformer, input_tokenizer, target_tokenizer, max_length_inp, max_length_targ)
            
            
            