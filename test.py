import json
import os

import tensorflow as tf

import data_loader as dl
import model.model_base as mb
import model.model_rl as mr
from framework import framework

FLAGS = tf.flags.FLAGS


def test_loader():
    return dl.json_file_data_loader(os.path.join(FLAGS.dataset_dir, 'test.json'),
                                    os.path.join(FLAGS.dataset_dir, 'word_vec.json'),
                                    os.path.join(FLAGS.dataset_dir, 'rel2id.json'),
                                    mode=dl.file_data_loader.MODE_ENTPAIR_BAG,
                                    shuffle=False)


if __name__ == '__main__':
    mb.init(is_training=False)
    model = mr.model_rl if 'rl' in FLAGS.se else mb.model

    fw = framework(test_data_loader=test_loader())
    with tf.variable_scope(FLAGS.model_name, reuse=tf.AUTO_REUSE):
        auc, pred_result = fw.test(model, model_name=FLAGS.model_name, return_result=True)
    with open(FLAGS.test_result_dir + FLAGS.model_name + "_pred.json", 'w') as of:
        json.dump(pred_result, of)
