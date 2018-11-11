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
        return tf.layers.conv1d(inputs=x, filters=hidden_size, kernel_size=kernel_size, strides=stride_size,
                                padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())


def cnn(x, mask=None, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu,
        var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or ('cnn' if mask is None else 'pcnn'), reuse=tf.AUTO_REUSE):
        cnn_cell = _cnn_cell(x, hidden_size, kernel_size, stride_size)
        pool = _pooling(cnn_cell) if mask is None else _piecewise_pooling(cnn_cell, mask)

        return dropout(activation(pool), keep_prob)


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
            bw_states, fw_states = birnn_states(cell_name, hidden_size, length, x)
            return tf.concat([fw_states, bw_states], axis=1)
        else:
            cell = _rnn_cell(hidden_size, cell_name)
            _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic_rnn')
            if isinstance(states, tuple):
                states = states[0]
            return states


def rcnn(x, length, rnn_hidden_size=230, cell_name='', bidirectional=False, mask=None, cnn_hidden_size=230,
         kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rcnn", reuse=tf.AUTO_REUSE):
        if bidirectional:
            fw_states, bw_states = birnn_states(x, length, rnn_hidden_size, cell_name)
            conv1 = _cnn_cell(tf.expand_dims(fw_states, 2), cnn_hidden_size)
            conv2 = _cnn_cell(tf.expand_dims(bw_states, 2), cnn_hidden_size)
            pool1 = _pooling(conv1) if mask is None else _piecewise_pooling(conv1, mask)
            pool2 = _pooling(conv2) if mask is None else _piecewise_pooling(conv2, mask)
            con1 = dropout(activation(pool1), keep_prob)
            con2 = dropout(activation(pool2), keep_prob)
            return tf.concat([con1, con2], -1)
        else:
            seq = rnn(x, length, rnn_hidden_size, cell_name, bidirectional, keep_prob=keep_prob)
            seq = tf.expand_dims(seq, 2)
            con = cnn(seq, mask, cnn_hidden_size, kernel_size, stride_size, activation, keep_prob=keep_prob)
            return con
