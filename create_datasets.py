from tqdm import tqdm
from os.path import join, exists

import numpy as np
import tensorflow as tf


def create_datasets(tokenizer, data_path, buffer_size, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    def create_dataset(read_path):
        print('Reading data from \"{}\"...'.format(read_path))

        with open(read_path, 'r') as f:
            lines = f.readlines()
            inputs = np.ones((len(lines), max_length), dtype = np.int32)
            labels = np.zeros(len(lines), dtype = np.int32)

            for i, line in tqdm(enumerate(lines), total = len(lines)):
                label, uttr = line.strip().split(' <SEP> ')
                uttr_ids = [SOS_ID] + tokenizer.encode(uttr)[:(max_length - 2)] + [EOS_ID]
                inputs[i,:len(uttr_ids)] = uttr_ids
                labels[i] = int(label)

            print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, labels

    train_inputs, train_labels = create_dataset(join(data_path, 'train.txt'))
    val_inputs, val_labels = create_dataset(join(data_path, 'valid.txt'))
    test_inputs, test_labels = create_dataset(join(data_path, 'test.txt'))

    train_inputs = np.concatenate([train_inputs, val_inputs])
    train_labels = np.concatenate([train_labels, val_labels])

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_inputs),
        tf.data.Dataset.from_tensor_slices(train_labels))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_inputs),
        tf.data.Dataset.from_tensor_slices(test_labels))

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return train_dataset, test_dataset