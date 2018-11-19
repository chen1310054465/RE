import tensorflow as tf

import framework as fw
import data_loader as dl
import model.model_base as mb
import model.model_rl as mr

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
    fw.init()
    model = mr.model_rl if '_rl' in FLAGS.se else mb.model

    framework = fw.framework(dl.json_file_data_loader(), dl.json_file_data_loader(dl.file_data_loader.TEST_PREFIX,
                                                                                  dl.file_data_loader.MODE_ENTPAIR_BAG,
                                                                                  shuffle=False))
    with tf.variable_scope(FLAGS.model_name, reuse=tf.AUTO_REUSE):
        framework.train(model)
