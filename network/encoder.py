import numpy as np
import tensorflow as tf

from framework import dropout


def _pooling(x):
    with tf.variable_scope("pooling", reuse=tf.AUTO_REUSE):
        return tf.reduce_max(x, axis=-2)


def _piecewise_pooling(x, mask):
    with tf.variable_scope("piecewise_pooling", reuse=tf.AUTO_REUSE):
        mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        mask = tf.nn.embedding_lookup(mask_embedding, mask)
        hidden_size = x.shape[-1]
        x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
        return tf.reshape(x, [-1, hidden_size * 3])


def _cnn_cell(x, hidden_size=230, kernel_size=3, stride_size=1):
    with tf.variable_scope("cnn_cell", reuse=tf.AUTO_REUSE):
        return tf.layers.conv1d(inputs=x,
                                filters=hidden_size,
                                kernel_size=kernel_size,
                                strides=stride_size,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer())


def cnn(x, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        x = _cnn_cell(x, hidden_size, kernel_size, stride_size)
        x = _pooling(x)
        x = activation(x)
        x = dropout(x, keep_prob)
        return x


def pcnn(x, mask, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        x = _cnn_cell(x, hidden_size, kernel_size, stride_size)
        x = _piecewise_pooling(x, mask)
        x = activation(x)
        x = dropout(x, keep_prob)
        return x


def _rnn_cell(hidden_size, cell_name='lstm'):
    with tf.variable_scope(cell_name, reuse=tf.AUTO_REUSE):
        if isinstance(cell_name, list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return _rnn_cell(hidden_size, cell_name[0])
            cells = [_rnn_cell(hidden_size, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(hidden_size)
        raise NotImplementedError


def rnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        x = dropout(x, keep_prob)
        cell = _rnn_cell(hidden_size, cell_name)
        _, states = tf.contrib.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic_rnn')
        if isinstance(states, tuple):
            states = states[0]
        return states


def birnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "birnn", reuse=tf.AUTO_REUSE):
        x = dropout(x, keep_prob)
        fw_cell = _rnn_cell(hidden_size, cell_name)
        bw_cell = _rnn_cell(hidden_size, cell_name)
        _, states = tf.contrib.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32,
                                                         scope='dynamic_bi_rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
        return tf.concat([fw_states, bw_states], axis=1)
