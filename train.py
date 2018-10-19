import os

import data_loader as dl
import model.model_base as mb
from framework import framework
from test import test_loader

# The first 3 parameters are train / test origin_data file name,
# word embedding file name and relation-id mapping file name respectively.
train_loader = dl.json_file_data_loader(os.path.join(mb.dataset_dir, 'train.json'),
                                        os.path.join(mb.dataset_dir, 'word_vec.json'),
                                        os.path.join(mb.dataset_dir, 'rel2id.json'),
                                        mode=dl.json_file_data_loader.MODE_RELFACT_BAG,
                                        shuffle=True)
fw = framework(train_loader, test_loader)

if __name__ == '__main__':
    mb.init()
    fw.train(mb.model, ckpt_dir="checkpoint",
             model_name=mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector,
             max_epoch=60, gpu_nums=1)
