import json
import numpy as np
import os

import tensorflow as tf

import framework as fw
import data_loader as dl
import model.model_base as mb
import model.model_rl as mr

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
    fw.init(is_training=False)
    model = mr.model_rl if 'rl' in FLAGS.se else mb.model

    framework = fw.framework(test_data_loader=dl.json_file_data_loader(dl.file_data_loader.TEST_PREFIX,
                                                                       dl.file_data_loader.MODE_ENTPAIR_BAG,
                                                                       shuffle=False))
    with tf.variable_scope(FLAGS.model_name, reuse=tf.AUTO_REUSE):
        auc, pred_result, output, acc_total, acc_not_na = framework.test(model, model_name=FLAGS.model_name,
                                                                         return_result=True)
    with open(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_pred.json"), 'w') as of:
        json.dump(pred_result, of)
    np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_out.npy"), output)
    with open(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_acc_total.json"), 'w') as of:
        json.dump(acc_total, of)
    with open(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_acc_not_na.json"), 'w') as of:
        json.dump(acc_not_na, of)
