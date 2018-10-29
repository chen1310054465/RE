import os
import sys

import numpy as np
import tensorflow as tf

from network import embedding, encoder, selector, classifier

FLAGS = tf.flags.FLAGS
# define some configurable parameter
tf.flags.DEFINE_string('dn', 'nyt', 'dataset name')
tf.flags.DEFINE_string('en', 'pcnn', 'encoder')
tf.flags.DEFINE_string('se', 'att', 'selector')
tf.flags.DEFINE_string('cl', 'softmax', 'classifier')
tf.flags.DEFINE_string('ac', 'relu', 'activation')
tf.flags.DEFINE_string('op', 'sgd', 'optimizer')
tf.flags.DEFINE_integer('ad', 0, 'adversarial training')
tf.flags.DEFINE_integer('gn', 1, 'gpu_nums')
tf.flags.DEFINE_string('pm', None, 'pretrain model')
# define some specified parameter
tf.flags.DEFINE_integer('max_epoch', 60, 'max epoch')
tf.flags.DEFINE_integer('save_epoch', 2, 'save epoch')
tf.flags.DEFINE_integer('hidden_size', 230, 'hidden size')
tf.flags.DEFINE_integer('batch_size', 160, 'batch size')
tf.flags.DEFINE_integer('max_length', 120, 'word max length')
tf.flags.DEFINE_float('learning_rate', 0.5, 'learning rate')
tf.flags.DEFINE_string('ckpt_dir', 'checkpoint', 'checkpoint dir')
tf.flags.DEFINE_string('summary_dir', 'summary', 'summary dir')
tf.flags.DEFINE_string('test_result_dir', 'test_result', 'test result dir')
tf.flags.DEFINE_string('processed_data_dir', 'processed_data', 'processed data dir')
tf.flags.DEFINE_string('dataset_dir', os.path.join('origin_data', FLAGS.dn), 'origin dataset dir')
tf.flags.DEFINE_string('model_name', (FLAGS.dn + '_' + FLAGS.en + "_" + FLAGS.se +  # dataset_name encoder selector
                                      (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                                      (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                                      (('_' + FLAGS.op) if FLAGS.op != 'sgd' else '') +  # optimizer
                                      ('_ad' if FLAGS.ad else '')), 'model_name')  # adversarial_training

activation = tf.nn.relu  # activation
optimizer = tf.train.GradientDescentOptimizer  # optimizer


def init(is_training=True):
    global activation, optimizer
    activations = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh,
                   'relu': tf.nn.relu, 'leaky_relu': tf.nn.leaky_relu}
    optimizers = {'sgd': tf.train.GradientDescentOptimizer, 'momentum': tf.train.MomentumOptimizer,
                  'adagrad': tf.train.AdagradOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
                  'adam': tf.train.AdamOptimizer}
    if 'help' in sys.argv:
        print('Usage: python3 ' + sys.argv[0] + ' [--dn dataset_name] [--en encoder] [--se selector] '
              + ('[--cl classifier] [--ac activation] [--op optimizer] [--ad adversarial_training] '
                 + '[--gn gpu_nums] [--pm pretrain_model]' if is_training else ''))
        print('*******************************args details******************************************')
        print('**  --dn: dataset_name: [nyt(New York Times dataset)]                              **')
        print('**  --en: encoder: [cnn pcnn rnn birnn rnn_gru birnn_gru]                          **')
        print('**  --se: selector: [instance att ave max att_rl ave_rl max_rl]                    **')
        if is_training:
            print('**  --cl: classifier: [softmax soft_label]                                         **')
            print('**  --ac: activation: ' + str([act for act in activations]) + '                    **')
            print('**  --op: optimizer: ' + str([op for op in optimizers]) + '            **')
            print('**  --ad: adversarial_training(whether add perturbation while training)            **')
            print('**  --gn: gpu_nums(denotes num of gpu for training)                                **')
            print('**  --pm: pretrain_model(denotes the name of model to pretrain, such as:pcnn_att)  **')
        print('*************************************************************************************')
        exit()

    if FLAGS.ac in activations:
        activation = activations[FLAGS.ac]

    if FLAGS.op in optimizers:
        optimizer = optimizers[FLAGS.op]


class model:
    def __init__(self, data_loader, is_training=True):
        self.rel_tot = data_loader.rel_tot
        self.word_vec = data_loader.word_vec
        self.is_training = is_training
        self.keep_prob = 0.5 if is_training else 1.0
        batch_size = FLAGS.batch_size // FLAGS.gn if is_training else FLAGS.batch_size

        self.word = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='pos2')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name="mask") \
            if 'pcnn' in FLAGS.en else None
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length') if 'rnn' in FLAGS.en else None
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label') if is_training else None
        self.instance_label = tf.placeholder(dtype=tf.int32, shape=[None], name='instance_label') \
            if 'att' or 'max' in FLAGS.se else None
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size + 1], name='scope') \
            if 'instance' not in FLAGS.se else None
        self.weights = tf.placeholder(dtype=tf.float32, shape=[batch_size], name='weights') if is_training else None

        self._network()

    def _network(self):
        # embedding
        self._embedding()
        with tf.variable_scope("re_" + FLAGS.en + "_" + FLAGS.se +
                               (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                               (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                               (('_' + FLAGS.op) if FLAGS.op != 'sgd' else ''),  # optimizer
                               reuse=tf.AUTO_REUSE):
            # encoder_selector_classifier
            self._encoder_selector_classifier(reuse=False if FLAGS.ad else True)
            # adversarial_training
            self._adversarial()

    def _encoder_selector_classifier(self, reuse=True):
        with tf.variable_scope(FLAGS.en + "_" + FLAGS.se +
                               (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else ''), reuse=reuse):
            self._encoder()  # encoder
            self._selector()  # selector
            self._classifier()  # classifier

    def _embedding(self):
        if not hasattr(self, 'embedding'):
            self.embedding = embedding.word_position_embedding(self.word, self.word_vec, self.pos1, self.pos2)

    def _encoder(self):
        if FLAGS.en == "pcnn":
            self.encoder = encoder.pcnn(self.embedding, self.mask, FLAGS.hidden_size, activation=activation,
                                        keep_prob=self.keep_prob)
        elif FLAGS.en == "cnn":
            self.encoder = encoder.cnn(self.embedding, FLAGS.hidden_size, activation=activation,
                                       keep_prob=self.keep_prob)
        elif "rnn" in FLAGS.en:
            ens = FLAGS.en.split('_')
            cell_name = ens[1] if len(ens) > 1 else "lstm"
            if ens[0] == "rnn":
                self.encoder = encoder.rnn(self.embedding, self.length, FLAGS.hidden_size, cell_name=cell_name,
                                           keep_prob=self.keep_prob)
            elif ens[0] == "birnn":
                self.encoder = encoder.birnn(self.embedding, self.length, FLAGS.hidden_size, cell_name=cell_name,
                                             keep_prob=self.keep_prob)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _selector(self):
        ses = FLAGS.se.split('_')
        se, rl = (ses[0], ses[1]) if len(ses) > 1 else (ses[0], None)
        if rl is not None and rl != 'rl':
            raise NotImplementedError
        if se == "att":
            self.logit, self.repre = selector.bag_attention(self.encoder, self.scope, self.instance_label,
                                                            self.rel_tot, self.is_training, keep_prob=self.keep_prob)
        elif se == "ave":
            self.logit, self.repre = selector.bag_average(self.encoder, self.scope, self.rel_tot,
                                                          self.is_training, keep_prob=self.keep_prob)
        elif se == "max":
            self.logit, self.repre = selector.bag_maximum(self.encoder, self.scope, self.instance_label,
                                                          self.rel_tot, self.is_training, keep_prob=self.keep_prob)
        elif se == "instance":
            self.logit, self.repre = selector.instance(self.encoder, self.rel_tot, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

    def _classifier(self):
        if self.is_training:
            if FLAGS.cl == "softmax":
                self.loss = classifier.softmax_cross_entropy(self.logit, self.label, self.rel_tot, weights=self.weights)
            elif FLAGS.cl == "soft_label":
                self.loss = classifier.soft_label_softmax_cross_entropy(self.logit, self.label, self.rel_tot
                                                                        , weights=self.weights)
            else:
                raise NotImplementedError
        self.output = classifier.output(self.logit)

    def _adversarial(self):
        if self.is_training and FLAGS.ad:
            with tf.variable_scope(FLAGS.en + '_' + FLAGS.se +
                                   (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +
                                   '_adversarial', reuse=tf.AUTO_REUSE):
                perturb = tf.gradients(self.loss, self.embedding)
                perturb = tf.reshape((0.01 * tf.stop_gradient(tf.nn.l2_normalize(perturb, dim=[0, 1, 2]))),
                                     [-1, FLAGS.max_length, self.embedding.shape[-1]])
                self.embedding = self.embedding + perturb
                self._encoder_selector_classifier()
