import numpy as np
import tensorflow as tf
# from keras_applications.resnet50 import identity_block, conv_block

from framework import dropout


def linear_transform(x, b):
    # w = tf.Variable(np.zeros(shape=[1, x.shape[-1]], dtype=np.float32), name='w')
    w = tf.get_variable('w', [1, x.shape[-1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    return w * x + b


def dense(inputs, units, activation=None, use_bias=True, kernel_initializer=None,
          bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None,
          activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True,
          name=None, reuse=None):
    return tf.layers.dense(inputs, units, activation, use_bias, kernel_initializer, bias_initializer,
                           kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                           bias_constraint, trainable, name, reuse)


def mask_embedding(num_piece):
    me = np.zeros([num_piece + 1, num_piece], dtype=np.float32)
    for i in range(num_piece):
        me[i + 1][i] = 1
    return tf.constant(me)


def _pooling(x):
    with tf.variable_scope("pooling", reuse=tf.AUTO_REUSE):
        return tf.reduce_max(x, axis=1)


def _piecewise_pooling(x, mask):
    with tf.variable_scope("piecewise_pooling", reuse=tf.AUTO_REUSE):
        # mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        mask = tf.nn.embedding_lookup(mask_embedding(num_piece=3), mask)
        hidden_size = x.shape[-1]
        x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
        return tf.reshape(x, [-1, hidden_size * 3])


def _cnn_cell(x, hidden_size=230, kernel_size=3, stride_size=1, activation=None, var_scope=None, is_2d=False):
    with tf.variable_scope(var_scope or "cnn_cell", reuse=tf.AUTO_REUSE):
        if is_2d:
            # x = tf.expand_dims(x, axis=1)
            x = tf.layers.conv2d(inputs=x, filters=hidden_size, kernel_size=[1, kernel_size], strides=[1, stride_size],
                                    padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        else:
            x = tf.layers.conv1d(inputs=x, filters=hidden_size, kernel_size=kernel_size, strides=stride_size,
                                    padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        return x if activation is None else activation(x)


def cnn(x, mask=None, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu,
        var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or ('cnn' if mask is None else 'pcnn'), reuse=tf.AUTO_REUSE):
        cnn_cell = _cnn_cell(x, hidden_size, kernel_size, stride_size)
        pool = _pooling(cnn_cell) if mask is None else _piecewise_pooling(cnn_cell, mask)

        return dropout(activation(pool), keep_prob)


def resnet(x, length=None, cell_name='lstm', bidirectional=False, mask=None, ib_num=4, hidden_size=230,
           kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or ('resnet' if mask is None else 'resnet_pcnn'), reuse=tf.AUTO_REUSE):
        seq = None if length is None else rnn(x, length, hidden_size, cell_name, bidirectional, keep_prob=keep_prob)
        x = _cnn_cell(x, hidden_size, kernel_size, stride_size, activation=activation)
        for i in range(ib_num):
            h1 = _cnn_cell(x, hidden_size, kernel_size, stride_size, activation=activation,
                           var_scope='conv_' + str(i) + 'a')
            h2 = _cnn_cell(h1, hidden_size, kernel_size, stride_size, activation=activation,
                           var_scope='conv_' + str(i) + 'b')
            x = h2 + x
        x = _pooling(x) if mask is None else _piecewise_pooling(x, mask)

        # x = conv_block(x, kernel_size, [hidden_size, hidden_size, 256], stage=2, block='a',
        #                strides=(stride_size, stride_size))
        # x = identity_block(x, kernel_size, [hidden_size, hidden_size, 256], stage=2, block='b')
        # x = identity_block(x, kernel_size, [hidden_size, hidden_size, 256], stage=2, block='c')
        x = dropout(activation(x), keep_prob)
        return x if seq is None else tf.concat([seq, x], axis=1)


def _rnn_cell(hidden_size, cell_name=''):
    with tf.variable_scope('BasicRNNCell' if cell_name.strip() == '' else cell_name, reuse=tf.AUTO_REUSE):
        if isinstance(cell_name, list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return _rnn_cell(hidden_size, cell_name[0])
            cells = [_rnn_cell(hidden_size, c) for c in cell_name]
            return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.strip() == '':
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_name.lower() == 'lstm':
            return tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        raise NotImplementedError


def birnn_states(x, length, hidden_size, cell_name):
    fw_cell = _rnn_cell(hidden_size, cell_name)
    bw_cell = _rnn_cell(hidden_size, cell_name)
    _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32,
                                                scope='dynamic_birnn')
    fw_states, bw_states = states
    if isinstance(fw_states, tuple):
        fw_states = fw_states[0]
        bw_states = bw_states[0]
    return fw_states, bw_states


def rnn(x, length, hidden_size=230, cell_name='', bidirectional=False, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or ('birnn' if bidirectional else 'rnn'), reuse=tf.AUTO_REUSE):
        x = dropout(x, keep_prob)
        if bidirectional:
            bw_states, fw_states = birnn_states(x, length, hidden_size, cell_name)
            return tf.concat([fw_states, bw_states], axis=1)
        else:
            cell = _rnn_cell(hidden_size, cell_name)
            _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic_rnn')
            if isinstance(states, tuple):
                states = states[0]
            return states


def rcnn(x, length, rnn_hidden_size=230, cell_name='', bidirectional=False, mask=None, cnn_hidden_size=230,
         kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0, li_encoder_mode=0):
    with tf.variable_scope(var_scope or "rcnn", reuse=tf.AUTO_REUSE):
        # if bidirectional:
        #     fw_states, bw_states = birnn_states(x, length, rnn_hidden_size, cell_name)
        #     conv1 = _cnn_cell(tf.expand_dims(fw_states, 2), cnn_hidden_size)
        #     conv2 = _cnn_cell(tf.expand_dims(bw_states, 2), cnn_hidden_size)
        #     pool1 = _pooling(conv1) if mask is None else _piecewise_pooling(conv1, mask)
        #     pool2 = _pooling(conv2) if mask is None else _piecewise_pooling(conv2, mask)
        #     con1 = dropout(activation(pool1), keep_prob)
        #     con2 = dropout(activation(pool2), keep_prob)
        #     return tf.concat([con1, con2], -1)
        # else:
        #     seq = rnn(x, length, rnn_hidden_size, cell_name, bidirectional, keep_prob=keep_prob)
        #     seq = tf.expand_dims(seq, 2)
        #     con = cnn(seq, mask, cnn_hidden_size, kernel_size, stride_size, activation, keep_prob=keep_prob)
        seq = rnn(x, length, rnn_hidden_size, cell_name, bidirectional, keep_prob=keep_prob)
        con = cnn(x, mask, cnn_hidden_size, kernel_size, stride_size, activation, keep_prob=keep_prob)
        if li_encoder_mode:
            return linear_transform(seq, tf.expand_dims(tf.reduce_sum(con, axis=-1), 1))
        else:
            return tf.concat([seq, con], axis=1)
