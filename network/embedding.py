import numpy as np
import tensorflow as tf


def concat(w_embedding, p_embedding, axis=-1):
    return tf.concat([w_embedding, p_embedding], axis=axis)


def word_embedding(word, word_vec, var_scope=None, word_embedding_dim=50, add_unk_and_blank=True):
    with tf.variable_scope(var_scope or 'word_embedding', reuse=tf.AUTO_REUSE):
        w_embedding = tf.get_variable('word_embedding', initializer=word_vec, dtype=tf.float32)
        if add_unk_and_blank:
            w_embedding = tf.concat([w_embedding,
                                     tf.get_variable("unk_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer()),
                                     tf.constant(np.zeros((1, word_embedding_dim), dtype=np.float32),
                                                 name='blank_word_embedding')], 0)
        x = tf.nn.embedding_lookup(w_embedding, word)
        return x


def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=120):
    with tf.variable_scope(var_scope or 'pos_embedding', reuse=tf.AUTO_REUSE):
        pos_tot = max_length * 2

        pos1_embedding = tf.get_variable('pos1_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        pos2_embedding = tf.get_variable('pos2_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
        # pos1_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), pos1_embedding], 0)
        # pos2_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), pos2_embedding], 0)
        input_pos1 = tf.nn.embedding_lookup(pos1_embedding, pos1)
        input_pos2 = tf.nn.embedding_lookup(pos2_embedding, pos2)
        x = tf.concat([input_pos1, input_pos2], -1)
        return x


def wp_embedding(word, word_vec, pos1, pos2, word_embedding_dim=50, pos_embedding_dim=5, max_length=120,
                 var_scope=None, add_unk_and_blank=True):
    with tf.variable_scope(var_scope or 'wp_embedding', reuse=tf.AUTO_REUSE):
        w_embedding = word_embedding(word, word_vec, var_scope=var_scope, word_embedding_dim=word_embedding_dim,
                                     add_unk_and_blank=add_unk_and_blank)
        p_embedding = pos_embedding(pos1, pos2, var_scope=var_scope, pos_embedding_dim=pos_embedding_dim,
                                    max_length=max_length)
        return w_embedding, p_embedding


def et_embedding(head_enttype, tail_enttype, enttype_tot, et_embedding_dim=12, var_scope=None, add_unk_and_blank=True):
    with tf.variable_scope(var_scope or 'et_embedding', reuse=tf.AUTO_REUSE):
        het_embedding = tf.get_variable('het_embedding', [enttype_tot, et_embedding_dim], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
        tet_embedding = tf.get_variable('tet_embedding', [enttype_tot, et_embedding_dim], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
        if add_unk_and_blank:
            het_embedding = tf.concat([het_embedding,
                                       tf.get_variable("unk_het_embedding", [1, et_embedding_dim], dtype=tf.float32,
                                                       initializer=tf.contrib.layers.xavier_initializer()),
                                       tf.constant(np.zeros((1, et_embedding_dim), dtype=np.float32),
                                                   name='blank_het_embedding')], 0)
            tet_embedding = tf.concat([tet_embedding,
                                       tf.get_variable("unk_tet_embedding", [1, et_embedding_dim], dtype=tf.float32,
                                                       initializer=tf.contrib.layers.xavier_initializer()),
                                       tf.constant(np.zeros((1, et_embedding_dim), dtype=np.float32),
                                                   name='blank_tet_embedding')], 0)
        input_head_enttype = tf.nn.embedding_lookup(het_embedding, head_enttype)
        input_tail_enttype = tf.nn.embedding_lookup(tet_embedding, tail_enttype)
        return input_head_enttype, input_tail_enttype
