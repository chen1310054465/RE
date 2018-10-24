import os

import tensorflow as tf

import data_loader as dl
import model.model_base as mb
import model.model_rl as mr
from framework import framework
from test import test_loader

FLAGS = tf.flags.FLAGS


# The first 3 parameters are train / test origin_data file name,
# word embedding file name and relation-id mapping file name respectively.
def train_loader():
    return dl.json_file_data_loader(os.path.join(FLAGS.dataset_dir, 'train.json'),
                                    os.path.join(FLAGS.dataset_dir, 'word_vec.json'),
                                    os.path.join(FLAGS.dataset_dir, 'rel2id.json'),
                                    mode=dl.json_file_data_loader.MODE_RELFACT_BAG,
                                    shuffle=True)


if __name__ == '__main__':
    mb.init()
    model = mr.model_rl if 'rl' in FLAGS.se else mb.model

    fw = framework(train_loader(), test_loader())
    with tf.variable_scope(FLAGS.model_name, reuse=tf.AUTO_REUSE):
        fw.train(model,  optimizer=mb.optimizer)
        if isinstance(model, mr.model_rl):
            fw.pretrain_policy_agent(model, optimizer=mb.optimizer, max_epoch=1)
            fw.train_rl(model, max_epoch=2)
