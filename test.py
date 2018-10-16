import json
import os

import data_loader as dl
from framework import re_framework
import model.model_base as mb

if __name__ == '__main__':
    test_loader = dl.json_file_data_loader(os.path.join(mb.dataset_dir, 'test.json'),
                                           os.path.join(mb.dataset_dir, 'word_vec.json'),
                                           os.path.join(mb.dataset_dir, 'rel2id.json'),
                                           mode=dl.json_file_data_loader.MODE_ENTPAIR_BAG,
                                           shuffle=False)

    fw = re_framework(test_data_loader=test_loader)
    mb.init()
    auc, pred_result = fw.test(mb.model,
                               ckpt="./checkpoint/" + mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector,
                               return_result=True)
    with open('./test_result/' + mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector + "_pred.json",
              'w') as outfile:
        json.dump(pred_result, outfile)
