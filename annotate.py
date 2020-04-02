# -*- coding: utf-8 -*-
"""Automatically annotating EmpatheticDialogues with BERT classifier
"""

import math
import tensorflow as tf
import csv
import numpy as np
from pytorch_transformers import RobertaTokenizer
import tqdm

from model import *
from utilities import *
from optimize import CustomSchedule
from create_datasets import create_datasets

tf.compat.v1.enable_eager_execution()

# After eager execution is enabled, operations are executed as they are
# defined and Tensor objects hold concrete values, which can be accessed as
# numpy.ndarray`s through the numpy() method.

emotions = ['afraid',
            'angry',
            'annoyed',
            'anticipating',
            'anxious',
            'apprehensive',
            'ashamed',
            'caring',
            'confident',
            'content',
            'devastated',
            'disappointed',
            'disgusted',
            'embarrassed',
            'excited',
            'faithful',
            'furious',
            'grateful',
            'guilty',
            'hopeful',
            'impressed',
            'jealous',
            'joyful',
            'lonely',
            'nostalgic',
            'prepared',
            'proud',
            'sad',
            'sentimental',
            'surprised',
            'terrified',
            'trusting']

labels = ['afraid', 
            'angry',
            'annoyed',
            'anticipating',
            'anxious',
            'apprehensive',
            'ashamed',
            'caring',
            'confident',
            'content',
            'devastated',
            'disappointed',
            'disgusted',
            'embarrassed',
            'excited',
            'faithful',
            'furious',
            'grateful',
            'guilty',
            'hopeful',
            'impressed',
            'jealous',
            'joyful',
            'lonely',
            'nostalgic',
            'prepared',
            'proud',
            'sad',
            'sentimental',
            'surprised',
            'terrified',
            'trusting',
            'agreeing',
            'acknowledging',
            'encouraging',
            'consoling',
            'sympathizing',
            'suggesting',
            'questioning',
            'wishing',
            'neutral']


num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 41  # Number of emotion categories
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size
max_length = 100  # Maximum number of tokens
buffer_size = 100000
batch_size = 1
num_epochs = 10
peak_lr = 2e-5
total_steps = 7000
warmup_steps = 700
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

checkpoint_path = './checkpoints'

SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
            layer_norm_eps, max_position_embed, vocab_size, num_emotions)

build_model(emobert, max_length, vocab_size)

learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
            epsilon = adam_epsilon)
#train_loss = tf.keras.metrics.Mean(name = 'train_loss')

# Define the checkpoint manager.
ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

# Restore the checkpoint at epoch 5 (checkpoint with highest accuracy on test set)
#print(ckpt_manager.checkpoints[4])
ckpt.restore(ckpt_manager.checkpoints[4])
print('Checkpoint at epoch 5 restored!!')


def predict_emotion(uttrs):

    bs = 1
    
    #with open(join(data_xpath, 'uttrs.txt'), 'r') as f:
    #    uttrs = f.read().splitlines()

    uttr_ids = np.ones((len(uttrs), max_length), dtype = np.int32)
    #for i, u in tqdm(enumerate(uttrs), total = len(uttrs)):
    i = 0
    u = uttrs[0]
    u_ids = [SOS_ID] + tokenizer.encode(u)[:(max_length-2)] + [EOS_ID]
    uttr_ids[i, :len(u_ids)] = u_ids

    uttr_emots = np.zeros((len(uttrs), num_emotions))
    num_batches = len(uttrs) // bs
    #for i in tqdm(range(num_batches)):
    i = 0
    s = i * bs
    t = s + bs
    inp = tf.constant(uttr_ids[s:t])
    enc_padding_mask = create_masks(inp)
    pred = emobert(inp, False, enc_padding_mask)
    pred = tf.nn.softmax(pred).numpy()

    return pred[0]
    #np.save(join(data_path, 'uttr_emots.npy'), uttr_emots)

for i in range(0,len(emotions)):

  emotion = emotions[i]
  print("Annotating dialogues of emotion: " + emotion + " ...")

  with open('./datasets/empatheticdialogues_unannotated/'+emotion+'.csv') as infile:
    
    with open('./datasets/empatheticdialogues_annotated/'+emotion+'.csv', 'a') as outfile:
      
      writer = csv.writer(outfile, delimiter=str(','), lineterminator='\n')
    
      readCSV = csv.reader(infile, delimiter=',')

      writable_row = ['Dialog_ID', 'Type', 'Actor', 'Text', 'Label']
      writer.writerow(writable_row)
      
      count = 0
      
      for row in readCSV:
        
        if count >= 1:

          writable_row = [row[0],row[1],row[2],row[3]]

          text = row[3]
          text = text.strip()
          
          predictions = predict_emotion([text])
          predictions = np.array(predictions)
          indices = predictions.argsort()[-1:][::-1]
          writable_row.append(labels[indices[0]])
          
          writer.writerow(writable_row)
          
        count += 1