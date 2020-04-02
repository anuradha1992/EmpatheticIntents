import numpy as np
import tensorflow as tf

# Masking
def create_padding_mask(seq):
    # To be consistent with RoBERTa, the padding index is set to 1.
    seq = tf.cast(tf.math.equal(seq, 1), tf.float32)

    # Add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp):
    enc_padding_mask = create_padding_mask(inp)
    return enc_padding_mask

def build_model(model, max_length, vocab_size):
    inp = np.ones((1, max_length), dtype = np.int32)
    inp[0,:max_length//2] = np.random.randint(2, vocab_size, size = max_length//2)
    inp = tf.constant(inp)
    enc_padding_mask = create_masks(inp)
    _ = model(inp, True, enc_padding_mask)
