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
tf.flags.DEFINE_integer('ad', 0, 'adversarial training')
tf.flags.DEFINE_integer('gn', 1, 'gpu_nums')
FLAGS = tf.flags.FLAGS
dataset_dir = os.path.join('origin_data', FLAGS.dn)
optimizer = tf.train.GradientDescentOptimizer
activation = tf.nn.relu


def init(is_training=True):
    activations = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh,
                   'relu': tf.nn.relu, 'leaky_relu': tf.nn.leaky_relu}
    optimizers = {'sgd': tf.train.GradientDescentOptimizer, 'momentum': tf.train.MomentumOptimizer,
                  'adagrad': tf.train.AdagradOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
                  'adam': tf.train.AdamOptimizer}
    if 'help' in sys.argv:
        print('Usage: python3 ' + sys.argv[0] + ' [--dn dataset_name] [--en encoder] [--se selector] '
              + ('[--cl classifier] [--ac activation] [--op optimizer] [--ad adversarial_training] '
                 + '[--gn gpu_nums]' if is_training else ''))
        print('*******************************args details******************************************')
        print('**  --dn: dataset_name: [nyt(New York Times dataset)]                              **')
        print('**  --en: encoder: [cnn pcnn rnn birnn]                                            **')
        print('**  --se: selector: [att ave max rl]                                               **')
        if is_training:
            print('**  --cl: classifier: [softmax soft_label]                                         **')
            print('**  --ac: activation: ' + str([act for act in activations]) + '                    **')
            print('**  --op: optimizer: ' + str([op for op in optimizers]) + '            **')
            print('**  --ad: adversarial_training(whether add perturbation while training)            **')
            print('**  --gn: gpu_nums(denotes num of gpu for training)                                **')
        print('*************************************************************************************')
        exit()

    if FLAGS.ac in activations:
        model.activation = activations[FLAGS.ac]

    if FLAGS.op in optimizers:
        model.optimizer = optimizers[FLAGS.op]


class model:
    def __init__(self, data_loader, batch_size, max_len=120, is_training=True):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_len], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_len], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_len], name='pos2')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_len], name="mask") if "pcnn" in FLAGS.en else None
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.instance_label = tf.placeholder(dtype=tf.int32, shape=[None], name='instance_label')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.data_loader = data_loader
        self.is_training = is_training
        self.keep_prob = 0.5 if is_training else 1.0

        # embedding
        wp_embedding = self._embedding()
        # encoder_selector_classifier
        self._encoder_selector_classifier(wp_embedding, reuse=False if FLAGS.ad else True)
        # adversarial_training
        self._adversarial(wp_embedding, max_len, FLAGS.ad)

    def _encoder_selector_classifier(self, wp_embedding, reuse=True):
        with tf.variable_scope(FLAGS.en + "_" + FLAGS.se +
                               (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                               (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                               (('_' + FLAGS.op) if FLAGS.op != 'sgd' else ''),  # optimizer
                               reuse=reuse):
            x = self._encoder(wp_embedding)  # encoder
            self._selector(x)  # selector
            self._classifier()  # classifier

    def _embedding(self):
        return embedding.word_position_embedding(self.word, self.data_loader.word_vec, self.pos1, self.pos2)

    def _encoder(self, wp_embedding):
        if FLAGS.en == "pcnn":
            x = encoder.pcnn(wp_embedding, self.mask, activation=activation, keep_prob=self.keep_prob)
        elif FLAGS.en == "cnn":
            x = encoder.cnn(wp_embedding, activation=activation, keep_prob=self.keep_prob)
        elif FLAGS.en == "rnn":
            x = encoder.rnn(wp_embedding, self.length, keep_prob=self.keep_prob)
        elif FLAGS.en == "birnn":
            x = encoder.birnn(wp_embedding, self.length, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError
        return x

    def _selector(self, x):
        if FLAGS.se == "att":
            self._logit, self._repre = selector.bag_attention(x, self.scope, self.instance_label,
                                                              self.data_loader.rel_tot, self.is_training,
                                                              keep_prob=self.keep_prob)
        elif FLAGS.se == "ave":
            self._logit, self._repre = selector.bag_average(x, self.scope, self.data_loader.rel_tot, self.is_training,
                                                            keep_prob=self.keep_prob)
        elif FLAGS.se == "max":
            self._logit, self._repre = selector.bag_maximum(x, self.scope, self.instance_label,
                                                            self.data_loader.rel_tot, self.is_training,
                                                            keep_prob=self.keep_prob)
        elif FLAGS.se == "instance":
            self._logit, self._repre = selector.instance(x, self.data_loader.rel_tot, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

    def _classifier(self):
        if self.is_training:
            if FLAGS.cl == "softmax":
                self._loss = classifier.softmax_cross_entropy(self._logit, self.label, self.data_loader.rel_tot,
                                                              weights_table=self._get_weights_table())
            elif FLAGS.cl == "soft_label":
                self._loss = classifier.soft_label_softmax_cross_entropy(self._logit, self.label,
                                                                         self.data_loader.rel_tot,
                                                                         weights_table=self._get_weights_table())
            else:
                raise NotImplementedError

    def _adversarial(self, wp_embedding, max_len, add_adversarial):
        if add_adversarial:
            perturb = tf.gradients(self._loss, wp_embedding)
            perturb = tf.reshape((0.01 * tf.stop_gradient(tf.nn.l2_normalize(perturb, dim=[0, 1, 2]))),
                                 [-1, max_len, wp_embedding.shape[-1]])
            self._encoder_selector_classifier(wp_embedding + perturb)

    def _get_weights_table(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros(self.data_loader.rel_tot, dtype=np.float32)
            for i in range(len(self.data_loader.data_label)):
                _weights_table[self.data_loader.data_label[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05 + 1e-20)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table

    def loss(self):
        return self._loss

    def logit(self):
        return self._logit

    def repre(self):
        return self._repre
