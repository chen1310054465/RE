import tensorflow as tf
from framework import dropout


def _logit(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        logit = tf.matmul(x, tf.transpose(relation_matrix)) + bias
    return logit


def _attention_train_logit(x, instance_label, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        current_relation = tf.nn.embedding_lookup(relation_matrix, instance_label)
        attention_logit = tf.reduce_sum(current_relation * x, -1)  # sum[(n', hidden_size) \dot (n', hidden_size)] = (n)
    return attention_logit


def _attention_test_logit(x, rel_tot, var_scope=None):
    with tf.variable_scope(var_scope or 'logit', reuse=tf.AUTO_REUSE):
        relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        # (n', hidden_size) x (hidden_size, rel_tot) = (n', rel_tot)
        attention_logit = tf.matmul(x, tf.transpose(relation_matrix))
    return attention_logit


def instance(x, rel_tot, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "instance", reuse=tf.AUTO_REUSE):
        x = dropout(x, keep_prob)
        return _logit(x, rel_tot), x


def bag_attention(x, scope, instance_label, rel_tot, is_training, var_scope=None, dropout_before=False, keep_prob=1.0):
    with tf.variable_scope(var_scope or "bag_attention", reuse=tf.AUTO_REUSE):
        if is_training:  # training
            if dropout_before:
                x = dropout(x, keep_prob)
            bag_repre = []
            attention_logit = _attention_train_logit(x, instance_label, rel_tot)
            for i in range(scope.shape[0] - 1):
                bag_hidden_mat = x[scope[i]:scope[i + 1]]
                attention_score = tf.nn.softmax(attention_logit[scope[i]:scope[i + 1]], -1)
                # (1, n') x (n', hidden_size) = (1, hidden_size) -> (hidden_size)
                bag_repre.append(tf.squeeze(tf.matmul(tf.expand_dims(attention_score, 0), bag_hidden_mat)))
            bag_repre = tf.stack(bag_repre)
            if not dropout_before:
                bag_repre = dropout(bag_repre, keep_prob)
            return _logit(bag_repre, rel_tot), bag_repre
        else:  # testing
            attention_logit = _attention_test_logit(x, rel_tot)  # (n, rel_tot)
            bag_repre = []
            bag_logit = []
            for i in range(scope.shape[0] - 1):
                bag_hidden_mat = x[scope[i]:scope[i + 1]]
                attention_score = tf.nn.softmax(tf.transpose(attention_logit[scope[i]:scope[i + 1], :]),
                                                -1)  # softmax of (rel_tot, n')
                bag_repre_for_each_rel = tf.matmul(attention_score,
                                                   bag_hidden_mat)  # (rel_tot, n') \dot (n', hidden_size) = (rel_tot, hidden_size)
                bag_logit_for_each_rel = _logit(bag_repre_for_each_rel, rel_tot)  # -> (rel_tot, rel_tot)
                bag_repre.append(bag_repre_for_each_rel)
                bag_logit.append(
                    tf.diag_part(tf.nn.softmax(bag_logit_for_each_rel, -1)))  # could be improved by sigmoid?
            bag_repre = tf.stack(bag_repre)
            bag_logit = tf.stack(bag_logit)
            return bag_logit, bag_repre


# could be improved?
def bag_one(x, scope, label, rel_tot, is_training, var_scope=None, dropout_before=False, keep_prob=1.0):
    with tf.variable_scope(var_scope or "maximum", reuse=tf.AUTO_REUSE):
        if is_training:  # training
            if dropout_before:
                x = dropout(x, keep_prob)
            bag_repre = []
            for i in range(scope.shape[0] - 1):
                bag_hidden_mat = x[scope[i]:scope[i + 1]]
                instance_logit = tf.nn.softmax(_logit(bag_hidden_mat, rel_tot), -1)  # (n', hidden_size)->(n', rel_tot)
                j = tf.argmax(instance_logit[:, label[i]], output_type=tf.int32)
                bag_repre.append(bag_hidden_mat[j])
            bag_repre = tf.stack(bag_repre)
            if not dropout_before:
                bag_repre = dropout(bag_repre, keep_prob)
            return _logit(bag_repre, rel_tot), bag_repre
        else:  # testing
            if dropout_before:
                x = dropout(x, keep_prob)
            bag_repre = []
            bag_logit = []
            for i in range(scope.shape[0] - 1):
                bag_hidden_mat = x[scope[i]:scope[i + 1]]
                instance_logit = tf.nn.softmax(_logit(bag_hidden_mat, rel_tot), -1)  # (n', hidden_size)->(n', rel_tot)
                bag_logit.append(tf.reduce_max(instance_logit, 0))
                bag_repre.append(bag_hidden_mat[0])  # fake max repre
            bag_logit = tf.stack(bag_logit)
            bag_repre = tf.stack(bag_repre)

            return tf.nn.softmax(bag_logit), bag_repre


def bag_average(x, scope, rel_tot, is_training, var_scope=None, dropout_before=False, keep_prob=1.0):
    with tf.variable_scope(var_scope or "average", reuse=tf.AUTO_REUSE):
        if dropout_before:
            x = dropout(x, keep_prob)
        bag_repre = []
        for i in range(scope.shape[0] - 1):
            bag_hidden_mat = x[scope[i]:scope[i + 1]]
            bag_repre.append(tf.reduce_mean(bag_hidden_mat, 0))  # (n', hidden_size) -> (hidden_size)
        bag_repre = tf.stack(bag_repre)
        if not dropout_before:
            bag_repre = dropout(bag_repre, keep_prob)

        bag_logit = _logit(bag_repre, rel_tot)
        if not is_training:
            bag_logit = tf.nn.softmax(bag_logit)
    return bag_logit, bag_repre


def bag_cross_max(x, scope, rel_tot, is_training, var_scope=None, dropout_before=False, keep_prob=1.0):
    """
    Cross-sentence Max-pooling proposed by (Jiang et al. 2016.)
    "Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks"
    https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf
    """
    with tf.variable_scope(var_scope or "cross_max", reuse=tf.AUTO_REUSE):
        if dropout_before:
            x = dropout(x, keep_prob)
        bag_repre = []
        for i in range(scope.shape[0] - 1):
            bag_hidden_mat = x[scope[i]:scope[i + 1]]
            bag_repre.append(tf.reduce_max(bag_hidden_mat, 0))  # (n', hidden_size) -> (hidden_size)
        bag_repre = tf.stack(bag_repre)
        if not dropout_before:
            bag_repre = dropout(bag_repre, keep_prob)
        bag_logit = _logit(bag_repre, rel_tot)
        if not is_training:
            bag_logit = tf.nn.softmax(bag_logit)
    return bag_logit, bag_repre
