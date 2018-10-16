import os

import data_loader as dl
from framework import re_framework
import model.model_base as mb

if __name__ == '__main__':
    # The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
    train_loader = dl.json_file_data_loader(os.path.join(mb.dataset_dir, 'train.json'),
                                            os.path.join(mb.dataset_dir, 'word_vec.json'),
                                            os.path.join(mb.dataset_dir, 'rel2id.json'),
                                            mode=dl.json_file_data_loader.MODE_RELFACT_BAG,
                                            shuffle=True)
    fw = re_framework(train_data_loader=train_loader)
    mb.init()
    fw.train(mb.model, ckpt_dir="checkpoint",
             model_name=mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector,
             max_epoch=60, gpu_nums=1)
