import os
import sys

import numpy as np
import tensorflow as tf

import data_loader as dl
from framework import framework
from network import embedding, encoder, selector, classifier


class model_base:
    def __init__(self, word_vec, rel_tot, batch_size, max_length=120):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.word_vec = word_vec
        self.rel_tot = rel_tot

    def loss(self):
        raise NotImplementedError

    def logit(self):
        raise NotImplementedError


class model(model_base):
    encoder = "pcnn"
    selector = "att"
    classifier = "softmax"

    def __init__(self, data_loader, batch_size, max_length=120, is_training=True):
        model_base.__init__(self, data_loader.word_vec, data_loader.rel_tot, batch_size, max_length)
        self.is_training = is_training
        self.keep_prob = 1.0
        if is_training:
            self.keep_prob = 0.5

        # Embedding
        wp_embedding = embedding.word_position_embedding(self.word, self.word_vec, self.pos1, self.pos2)

        # Encoder
        if model.encoder == "pcnn":
            self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
            x = encoder.pcnn(wp_embedding, self.mask, keep_prob=self.keep_prob)
        elif model.encoder == "cnn":
            x = encoder.cnn(wp_embedding, keep_prob=self.keep_prob)
        elif model.encoder == "rnn":
            x = encoder.rnn(wp_embedding, self.length, keep_prob=self.keep_prob)
        elif model.encoder == "birnn":
            x = encoder.birnn(wp_embedding, self.length, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

        # Selector
        if model.selector == "att":
            self._logit, self._repre = selector.bag_attention(x, self.scope, self.ins_label,
                                                              self.rel_tot, is_training, keep_prob=self.keep_prob)
        elif model.selector == "ave":
            self._logit, self._repre = selector.bag_average(x, self.scope, self.rel_tot, is_training,
                                                            keep_prob=self.keep_prob)
        elif model.selector == "max":
            self._logit, self._repre = selector.bag_maximum(x, self.scope, self.ins_label,
                                                            self.rel_tot, is_training, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

        if is_training:
            # Classifier
            if model.classifier == "softmax":
                self._loss = classifier.softmax_cross_entropy(self._logit, self.label, self.rel_tot,
                                                              weights_table=self.get_weights_table(data_loader))
            elif model.classifier == "soft_label":
                self._loss = classifier.soft_label_softmax_cross_entropy(self._logit, self.label, self.rel_tot,
                                                                         weights_table=self.get_weights_table(
                                                                             data_loader))
            else:
                raise NotImplementedError

    def loss(self):
        return self._loss

    def logit(self):
        return self._logit

    def repre(self):
        return self._repre

    def get_weights_table(self, data_loader):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros(self.rel_tot, dtype=np.float32)
            for i in range(len(data_loader.data_rel)):
                _weights_table[data_loader.data_rel[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05 + 1e-20)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table


dataset_name = 'nyt'
dataset_dir = os.path.join('data', dataset_name)
fw = None


def init():
    global dataset_name, dataset_dir, fw
    # The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
    train_loader = dl.json_file_data_loader(os.path.join(dataset_dir, 'train.json'),
                                            os.path.join(dataset_dir, 'word_vec.json'),
                                            os.path.join(dataset_dir, 'rel2id.json'),
                                            mode=dl.json_file_data_loader.MODE_RELFACT_BAG,
                                            shuffle=True)
    test_loader = dl.json_file_data_loader(os.path.join(dataset_dir, 'test.json'),
                                           os.path.join(dataset_dir, 'word_vec.json'),
                                           os.path.join(dataset_dir, 'rel2id.json'),
                                           mode=dl.json_file_data_loader.MODE_ENTPAIR_BAG,
                                           shuffle=False)
    fw = framework(train_loader, test_loader)

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        dataset_dir = os.path.join('data', dataset_name)
    if not os.path.isdir(dataset_dir):
        raise Exception("[ERROR] Dataset dir %s doesn't exist!" % dataset_dir)
    if len(sys.argv) > 2:
        model.encoder = sys.argv[2]
    if len(sys.argv) > 3:
        model.selector = sys.argv[3]
    if len(sys.argv) > 4:
        model.classifier = sys.argv[4]
