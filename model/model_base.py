import os
import sys

import numpy as np
import tensorflow as tf

from network import embedding, encoder, selector, classifier

# define some parameter
tf.flags.DEFINE_string('dn', 'nyt', 'dataset_name')
tf.flags.DEFINE_string('en', 'pcnn', 'encoder')
tf.flags.DEFINE_string('se', 'att', 'selector')
tf.flags.DEFINE_string('cl', 'softmax', 'classifier')
tf.flags.DEFINE_string('ac', 'relu', 'activation')
tf.flags.DEFINE_string('op', 'sgd', 'optimizer')
tf.flags.DEFINE_integer('gn', 1, 'gpu_nums')
FLAGS = tf.flags.FLAGS
dataset_dir = os.path.join('origin_data', FLAGS.dn)
optimizer = tf.train.GradientDescentOptimizer
activation = tf.nn.relu


def init():
    activations = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh,
                   'relu': tf.nn.relu, 'leaky_relu': tf.nn.leaky_relu}
    optimizers = {'sgd': tf.train.GradientDescentOptimizer, 'momentum': tf.train.MomentumOptimizer,
                  'adagrad': tf.train.AdagradOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
                  'adam': tf.train.AdamOptimizer}
    if 'help' in sys.argv:
        print('Usage: python3 ' + sys.argv[0] + ' [--dn dataset_name] [--en encoder] '
              + '[--se selector] [--cl classifier] [--ac activation] '
              + '[--op optimizer] [--gn gpu_nums]')
        print('*******************************args details******************************************')
        print('**  --dn: dataset_name(nyt: New York Times dataset)                                **')
        print('**  --en: encoder(such as: cnn pcnn rnn birnn)                                     **')
        print('**  --se: selector(such as: att ave max)                                           **')
        print('**  --cl: classifier(such as: softmax soft_label)                                  **')
        print('**  --ac: activation(such as: ' + str([act for act in activations]) + ')           **')
        print('**  --op: optimizer(such as: ' + str([op for op in optimizers]) + ')   **')
        print('**  --gn: gpu_nums(denotes num of gpu for training)                                **')
        print('*************************************************************************************')
        exit()

    if FLAGS.ac in activations:
        model.activation = activations[FLAGS.ac]

    if FLAGS.op in optimizers:
        model.optimizer = optimizers[FLAGS.op]


class model_base:
    def __init__(self, word_vec, rel_tot, batch_size, max_length=120):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.instance_label = tf.placeholder(dtype=tf.int32, shape=[None], name='instance_label')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.word_vec = word_vec
        self.rel_tot = rel_tot

    def loss(self):
        raise NotImplementedError

    def logit(self):
        raise NotImplementedError


class model(model_base):
    def __init__(self, data_loader, batch_size, max_length=120, is_training=True):
        model_base.__init__(self, data_loader.word_vec, data_loader.rel_tot, batch_size, max_length)
        self.is_training = is_training
        self.keep_prob = 1.0
        if is_training:
            self.keep_prob = 0.5

        # embedding
        wp_embedding = embedding.word_position_embedding(self.word, self.word_vec, self.pos1, self.pos2)

        # encoder
        if FLAGS.en == "pcnn":
            self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
            x = encoder.pcnn(wp_embedding, self.mask, activation=activation, keep_prob=self.keep_prob)
        elif FLAGS.en == "cnn":
            x = encoder.cnn(wp_embedding, activation=activation, keep_prob=self.keep_prob)
        elif FLAGS.en == "rnn":
            x = encoder.rnn(wp_embedding, self.length, keep_prob=self.keep_prob)
        elif FLAGS.en == "birnn":
            x = encoder.birnn(wp_embedding, self.length, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

        # selector
        if FLAGS.se == "att":
            self._logit, self._repre = selector.bag_attention(x, self.scope, self.instance_label,
                                                              self.rel_tot, is_training, keep_prob=self.keep_prob)
        elif FLAGS.se == "ave":
            self._logit, self._repre = selector.bag_average(x, self.scope, self.rel_tot, is_training,
                                                            keep_prob=self.keep_prob)
        elif FLAGS.se == "max":
            self._logit, self._repre = selector.bag_maximum(x, self.scope, self.instance_label,
                                                            self.rel_tot, is_training, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

        if is_training:
            # classifier
            if FLAGS.cl == "softmax":
                self._loss = classifier.softmax_cross_entropy(self._logit, self.label, self.rel_tot,
                                                              weights_table=self.get_weights_table(data_loader))
            elif FLAGS.cl == "soft_label":
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
            for i in range(len(data_loader.data_label)):
                _weights_table[data_loader.data_label[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05 + 1e-20)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table
