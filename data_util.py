import os
import json
# import numpy as np

origin_data_dir = os.path.join('origin_data', 'nyt')
test_result_dir = os.path.join('test_result', 'nyt')


def acc_diff():
    # acc_total_fn = os.path.join(test_result_dir, 'nyt_pcnn_att_acc_total.json')
    # acc_total = json.load(open(acc_total_fn, 'r'))
    acc_not_na_fn = os.path.join(test_result_dir, 'nyt_pcnn_att_acc_not_na.json')
    acc_not_na = json.load(open(acc_not_na_fn, 'r'))
    # etp_acc_total_fn = os.path.join(test_result_dir, 'nyt_etp_pcnn_att_acc_total.json')
    # etp_acc_total = json.load(open(etp_acc_total_fn, 'r'))
    etp_acc_not_na_fn = os.path.join(test_result_dir, 'nyt_etp_pcnn_att_acc_not_na.json')
    etp_acc_not_na = json.load(open(etp_acc_not_na_fn, 'r'))
    return list(set(etp_acc_not_na['acc_idx']) - set(acc_not_na['acc_idx']))


def load_test_data():
    test_data_fn = os.path.join(origin_data_dir, 'test.json')
    test_data = json.load(open(test_data_fn, 'r'))
    test_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
    pre_entpair = None
    return_data = {}
    k = -1
    for instance in test_data:
        cur_entpair = instance['head']['id'] + '#' + instance['tail']['id']
        if cur_entpair != pre_entpair:
            k += 1
            pre_entpair = cur_entpair
            return_data[k] = [instance]
        else:
            return_data[k].append(instance)
    return return_data


if __name__ == '__main__':
    diff = acc_diff()
    data = load_test_data()
    out_data = {}
    for i in diff:
        out_data[i] = data[i]
    # np.save(os.path.join(test_result_dir, 'acc_diff.npy'), data[diff])
    with open(os.path.join(test_result_dir, 'acc_diff.json'), 'w') as of:
        json.dump(out_data, of, indent=4)
