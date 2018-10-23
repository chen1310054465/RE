import json
import os

import tensorflow as tf

import data_loader as dl
import model.model_base as mb
from framework import framework

FLAGS = tf.flags.FLAGS


def test_loader():
    return dl.json_file_data_loader(os.path.join(mb.dataset_dir, 'test.json'),
                                    os.path.join(mb.dataset_dir, 'word_vec.json'),
                                    os.path.join(mb.dataset_dir, 'rel2id.json'),
                                    mode=dl.json_file_data_loader.MODE_ENTPAIR_BAG,
                                    shuffle=False)


if __name__ == '__main__':
    mb.init(is_training=False)
    fw = framework(test_data_loader=test_loader())
    auc, pred_result = fw.test(mb.model, ckpt="./checkpoint/" + mb.dataset_dir.split(os.sep)[-1]  # dataset_name
                                              + '_' + FLAGS.en + "_" + FLAGS.se +  # encoder selector
                                              (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                                              (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                                              (('_' + FLAGS.op) if FLAGS.op != 'sgd' else '') +  # optimizer
                                              ('_ad' if FLAGS.ad else ''),  # adversarial_training
                               return_result=True)
    with open('./test_result/' + mb.dataset_dir.split(os.sep)[-1]
              + '_' + FLAGS.en + "_" + FLAGS.se +  # encoder selector
              (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
              (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
              (('_' + FLAGS.op) if FLAGS.op != 'sgd' else '') +  # optimizer
              ('_ad' if FLAGS.ad else '') + "_pred.json", 'w') as outfile:
        json.dump(pred_result, outfile)
