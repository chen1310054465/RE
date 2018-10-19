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
    mb.init()
    fw = framework(test_data_loader=test_loader())
    auc, pred_result = fw.test(mb.model,
                               ckpt="./checkpoint/" + mb.dataset_dir.split(os.sep)[-1] + "_" +
                                    FLAGS.en + "_" + FLAGS.se, return_result=True)
    with open('./test_result/' + mb.dataset_dir.split(os.sep)[-1]
              + "_" + FLAGS.en + "_" + FLAGS.se + "_pred.json", 'w') as outfile:
        json.dump(pred_result, outfile)
