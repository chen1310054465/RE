import os

import tensorflow as tf

import data_loader as dl
import model.model_base as mb
from framework import framework
from test import test_loader

FLAGS = tf.flags.FLAGS


# The first 3 parameters are train / test origin_data file name,
# word embedding file name and relation-id mapping file name respectively.
def train_loader():
    return dl.json_file_data_loader(os.path.join(mb.dataset_dir, 'train.json'),
                                    os.path.join(mb.dataset_dir, 'word_vec.json'),
                                    os.path.join(mb.dataset_dir, 'rel2id.json'),
                                    mode=dl.json_file_data_loader.MODE_RELFACT_BAG,
                                    shuffle=True)


if __name__ == '__main__':
    mb.init()
    fw = framework(train_loader(), test_loader())
    fw.train(mb.model, ckpt_dir="checkpoint",  # ckpt_dir
             model_name=mb.dataset_dir.split(os.sep)[-1]  # dataset_name
                        + '_' + FLAGS.en + "_" + FLAGS.se +  # encoder selector
                        (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                        (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                        (('_' + FLAGS.op) if FLAGS.op != 'sgd' else ''),  # optimizer
             max_epoch=60, optimizer=mb.optimizer, gpu_nums=FLAGS.gn)
