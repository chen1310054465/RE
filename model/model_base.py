import re

import tensorflow as tf

from network import embedding, encoder, selector, classifier

FLAGS = tf.flags.FLAGS


class model:
    def __init__(self, data_loader, activation=tf.nn.relu, is_training=True):
        self.rel_tot = data_loader.rel_tot
        self.enttype_tot = data_loader.enttype_tot
        self.word_vec = data_loader.word_vec
        self.activation = activation
        self.is_training = is_training
        self.keep_prob = 0.5 if is_training else 1.0
        batch_size = FLAGS.batch_size // FLAGS.gn if is_training else FLAGS.batch_size

        self.word = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name='pos2')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_length], name="mask") \
            if 'pcnn' in FLAGS.en else None
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length') \
            if re.search("r.*nn", FLAGS.en) else None
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label') \
            if is_training or 'one' in FLAGS.se else None
        self.instance_label = tf.placeholder(dtype=tf.int32, shape=[None], name='instance_label') \
            if 'att' in FLAGS.se else None
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size + 1], name='scope') \
            if 'instance' not in FLAGS.se else None
        self.weights = tf.placeholder(dtype=tf.float32, shape=[batch_size], name='weights') if is_training else None
        self.head_enttype = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.et_max_length], name="head_enttype") \
            if FLAGS.et else None
        self.tail_enttype = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.et_max_length], name="tail_enttype") \
            if FLAGS.et else None

        self._network()

    def _network(self):
        # embedding
        self._embedding()
        with tf.variable_scope(('et_' if FLAGS.et else '') +  # entity_type
                               FLAGS.en + "_" + FLAGS.se +
                               (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                               (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                               (('_' + FLAGS.op) if FLAGS.op != 'sgd' else '') +  # optimizer
                               '_ad_' + ('y' if FLAGS.ad != 0 else 'n'),  # adversarial
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
        if not hasattr(self, 'wp_embedding'):
            self.w_embedding, self.p_embedding = embedding.word_position_embedding(self.word, self.word_vec,
                                                                                   self.pos1, self.pos2)
            self.wp_embedding = embedding.concat(self.w_embedding, self.p_embedding)
        if FLAGS.et and not hasattr(self, 'et_embedding'):
            self.head_et_embedding, self.tail_et_embedding = embedding.ent_type_embedding(self.head_enttype,
                                                                                          self.tail_enttype,
                                                                                          self.enttype_tot,
                                                                                          et_embedding_dim=FLAGS.et_dim)
            self.et_embedding = embedding.concat(self.head_et_embedding, self.tail_et_embedding)

    def _encoder(self):
        if FLAGS.en == "cnn" or FLAGS.en == "pcnn":
            self.encoder = encoder.cnn(self.wp_embedding, self.mask, FLAGS.hidden_size,
                                       activation=self.activation, keep_prob=self.keep_prob)
        elif re.search("r.*nn", FLAGS.en):
            ens = FLAGS.en.split('_')
            cell_name = ens[1] if len(ens) > 1 else ""
            if ens[0] == "rnn" or ens[0] == "birnn":
                self.encoder = encoder.rnn(self.wp_embedding, self.length, FLAGS.hidden_size, cell_name=cell_name,
                                           bidirectional=ens[0] == "birnn", keep_prob=self.keep_prob)
            elif ens[0] == "rcnn" or ens[0] == "rpcnn" or ens[0] == "bircnn" or ens[0] == "birpcnn":
                self.encoder = encoder.rcnn(self.wp_embedding, self.length, rnn_hidden_size=FLAGS.rnn_hidden_size,
                                            cell_name=cell_name, bidirectional='bi' in ens[0],
                                            mask=self.mask, cnn_hidden_size=FLAGS.cnn_hidden_size,
                                            activation=self.activation, keep_prob=self.keep_prob)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if FLAGS.et:
            self.et_encoder = encoder.cnn(self.et_embedding, FLAGS.et_hidden_size, activation=self.activation,
                                          keep_prob=self.keep_prob)
            self.encoder = tf.concat([self.encoder, self.et_encoder], -1)

    def _selector(self):
        ses = FLAGS.se.split('_')
        se, rl = (ses[0], ses[1]) if len(ses) > 1 else (ses[0], None)
        if rl is not None and rl != 'rl':
            raise NotImplementedError
        if se == "att":
            self.logit, self.repre = selector.bag_attention(self.encoder, self.scope, self.instance_label,
                                                            self.rel_tot, self.is_training, keep_prob=self.keep_prob)
        elif se == "one":
            self.logit, self.repre = selector.bag_one(self.encoder, self.scope, self.label,
                                                      self.rel_tot, self.is_training, keep_prob=self.keep_prob)
        elif se == "ave":
            self.logit, self.repre = selector.bag_average(self.encoder, self.scope, self.rel_tot,
                                                          self.is_training, keep_prob=self.keep_prob)
        elif se == "cross_max":
            self.logit, self.repre = selector.bag_cross_max(self.encoder, self.scope, self.rel_tot,
                                                            self.is_training, keep_prob=self.keep_prob)
        elif se == "instance":
            self.logit, self.repre = selector.instance(self.encoder, self.rel_tot, keep_prob=self.keep_prob)
        else:
            raise NotImplementedError

    def _classifier(self):
        if self.is_training:
            if FLAGS.cl == "softmax":
                self.loss = classifier.softmax_cross_entropy(self.logit, self.label, self.rel_tot, weights=self.weights)
            elif FLAGS.cl == "sigmoid":
                self.loss = classifier.sigmoid_cross_entropy(self.logit, self.label, self.rel_tot, weights=self.weights)
            elif FLAGS.cl == "soft_label":
                self.loss = classifier.soft_label_softmax_cross_entropy(self.logit, self.label, self.rel_tot,
                                                                        weights=self.weights)
            else:
                raise NotImplementedError
        self.output = classifier.output(self.logit)

    def _adversarial(self):
        if self.is_training and FLAGS.ad:
            with tf.variable_scope(FLAGS.en + '_' + FLAGS.se +
                                   (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +
                                   '_adversarial', reuse=tf.AUTO_REUSE):
                perturb = tf.gradients(self.loss, self.w_embedding)
                perturb = tf.reshape((0.01 * tf.stop_gradient(tf.nn.l2_normalize(perturb, dim=[0, 1, 2]))),
                                     [-1, FLAGS.max_length, self.w_embedding.shape[-1]])
                self.w_embedding = self.w_embedding + perturb
                self.wp_embedding = embedding.concat(self.w_embedding, self.p_embedding)
            self._encoder_selector_classifier()
