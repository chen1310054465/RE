import tensorflow as tf


def softmax_cross_entropy(x, label, rel_tot, weights_table=None, var_scope=None):
    with tf.variable_scope(var_scope or "loss", reuse=tf.AUTO_REUSE):
        if weights_table is None:
            weights = 1.0
        else:
            weights = tf.nn.embedding_lookup(weights_table, label)
        label_onehot = tf.one_hot(indices=label, depth=rel_tot, dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=x, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss


# soft-label:  I just implemented it, but I haven't got the result in paper.
def soft_label_softmax_cross_entropy(x, label, weights, rel_tot):
    with tf.name_scope("soft-label-loss"):
        label_onehot = tf.one_hot(indices=label, depth=rel_tot, dtype=tf.int32)
        n_score = x + 0.9 * tf.reshape(tf.reduce_max(x, 1), [-1, 1]) * tf.cast(label_onehot, tf.float32)
        n_label = tf.one_hot(indices=tf.reshape(tf.argmax(n_score, axis=1), [-1]), depth=rel_tot,
                             dtype=tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=n_label, logits=n_score, weights=weights)
        tf.summary.scalar('loss', loss)
        return loss


def output(x):
    return tf.argmax(x, axis=-1)
