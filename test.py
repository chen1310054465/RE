import json

import model.model_base as mb

if __name__ == '__main__':
    mb.init()
    auc, pred_result = mb.fw.test(mb.model,
                                  ckpt="./checkpoint/" + mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector,
                                  return_result=True)
    with open('./test_result/' + mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector + "_pred.json",
              'w') as outfile:
        json.dump(pred_result, outfile)
