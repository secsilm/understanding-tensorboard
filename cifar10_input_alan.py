import os

import numpy as np
import tensorflow as tf

import _pickle as pickle

# 训练数据文件
TRAIN_FILE = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

# 验证数据文件
EVAL_FILE = ['test_batch']


def unpickle(filename):
    '''Decode the dataset files.'''
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        return d


def onehot(labels, cls=None):
    ''' One-hot encoding, zero-based'''
    n_sample = len(labels)
    if not cls:
        n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


def merge_data(dataset_dir, onehot_encoding=False):
    train_images = unpickle(os.path.join(dataset_dir, TRAIN_FILE[0]))['data']
    train_labels = unpickle(os.path.join(dataset_dir, TRAIN_FILE[0]))['labels']
    eval_images = unpickle(os.path.join(dataset_dir, EVAL_FILE[0]))['data']
    eval_labels = unpickle(os.path.join(dataset_dir, EVAL_FILE[0]))['labels']
    # 训练集
    for i in range(2, len(TRAIN_FILE) + 1):
        batch = unpickle(os.path.join(dataset_dir, TRAIN_FILE[i - 1]))
        train_images = np.concatenate((train_images, batch['data']), axis=0)
        train_labels = np.concatenate((train_labels, batch['labels']), axis=0)
    # 验证集
    for i in range(2, len(EVAL_FILE) + 1):
        batch = unpickle(os.path.join(dataset_dir, TRAIN_FILE[i - 1]))
        eval_images = np.concatenate((eval_images, batch['data']), axis=0)
        eval_labels = np.concatenate((eval_labels, batch['labels']), axis=0)
    if onehot_encoding:
        train_labels = onehot(train_labels)
        eval_labels = onehot(eval_labels)
    return train_images, eval_images, train_labels, eval_labels


class Cifar10(object):

    def __init__(self, images, lables):
        '''dataset_dir: the dir which saves the dataset files.
           onehot: if ont-hot encoding or not'''
        self._num_exzamples = len(lables)
        self._images = images
        self._labels = lables
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_exzamples(self):
        return self._num_exzamples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_exzamples:
            # 重新开始一个新的 epoch
            self._epochs_completed += 1
            # 重新打乱数据集
            idx = np.arange(self._num_exzamples)
            np.random.shuffle(idx)
            self._images = self._images[idx, :]
            self._labels = self._labels[idx, :]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end, :], self._labels[start:end, :]


def read_dataset(dataset_dir, onehot_encoding=False):
    class Datasets(object):
        pass
    dataset = Datasets()
    train_images, eval_images, train_labels, eval_labels = merge_data(dataset_dir, onehot_encoding)
    dataset.train = Cifar10(train_images, train_labels) 
    dataset.eval = Cifar10(eval_images, eval_labels)
    return dataset
