# -*- coding: utf-8 -*-
"""Training the BERT transformer based classifier
"""

import tensorflow as tf
import numpy as np
import time
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from model import *
from utilities import *
from optimize import CustomSchedule
from create_datasets import create_datasets


tf.compat.v1.enable_eager_execution()

# After eager execution is enabled, operations are executed as they are
# defined and Tensor objects hold concrete values, which can be accessed as
# numpy.ndarray`s through the numpy() method.


# hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 41  # Number of emotion + intent categories
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size
max_length = 100  # Maximum number of tokens
buffer_size = 100000
batch_size = 32
num_epochs = 10
peak_lr = 2e-5
total_steps = 7000
warmup_steps = 700
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

checkpoint_path = './checkpoints'
log_path = './log/emobert.log'
data_path = './datasets/train_data'

f = open(log_path, 'w', encoding = 'utf-8')

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    train_dataset, test_dataset = create_datasets(tokenizer, data_path, buffer_size,
        batch_size, max_length)
    train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

    # Define the model.
    emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_emotions)

    # Build the model and initialize weights from PlainTransformer pre-trained on OpenSubtitles.
    build_model(emobert, max_length, vocab_size)
    emobert.load_weights('./pretrained_weights/roberta2emobert.h5')
    print('Weights initialized from RoBERTa.')
    f.write('Weights initialized from RoBERTa.\n')

    # Define optimizer and metrics.
    learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')


    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
        f.write('Latest checkpoint restored!!\n')

    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            # inp.shape == (batch_size, seq_len)
            # tar_emot.shape == (batch_size,)
            inp, tar_emot = inputs
            enc_padding_mask = create_masks(inp)

            with tf.GradientTape() as tape:
                pred_emot = emobert(inp, True, enc_padding_mask)  # (batch_size, num_emotions)
                losses_per_examples = loss_function(tar_emot, pred_emot)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

            gradients = tape.gradient(loss, emobert.trainable_variables)    
            optimizer.apply_gradients(zip(gradients, emobert.trainable_variables))
            return loss

        losses_per_replica = mirrored_strategy.experimental_run_v2(
            step_fn, args = (dist_inputs,))
        mean_loss = mirrored_strategy.reduce(
            tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)

        train_loss(mean_loss)
        return mean_loss

    def validate():
        accuracy = []
        for (batch, inputs) in enumerate(test_dataset):
            inp, tar_emot = inputs
            enc_padding_mask = create_masks(inp)
            pred_emot = emobert(inp, False, enc_padding_mask)
            pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
            accuracy.append(np.mean(tar_emot.numpy() == pred_emot))
        return np.mean(accuracy)

    # Start training
    for epoch in range(num_epochs):
        start = time.time()

        train_loss.reset_states()

        for (batch, inputs) in enumerate(train_dataset):
            current_loss = train_step(inputs)
            current_mean_loss = train_loss.result()
            print('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}'.format(
                epoch + 1, batch, current_mean_loss, current_loss))
            f.write('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                epoch + 1, batch, current_mean_loss, current_loss))

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        f.write('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

        epoch_loss = train_loss.result()
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, epoch_loss))
        f.write('Epoch {} Loss {:.4f}\n'.format(epoch + 1, epoch_loss))

        current_time = time.time()
        print('Time taken for 1 epoch: {} secs'.format(current_time - start))
        f.write('Time taken for 1 epoch: {} secs\n'.format(current_time - start))

        val_ac = validate()
        print('Current accuracy on validation set: {}\n'.format(val_ac))
        f.write('Current accuracy on validation set: {}\n\n'.format(val_ac))

f.close()