import json
import os
import sys

import Document_pb2 as pb
from google.protobuf.internal.decoder import _DecodeVarint32

guid2entity = {}


def get_entities(file_name):
    print("Loading entities...")
    f = open(file_name, 'rb')
    for line in f.readlines():
        line = line.rstrip()
        guid, word, tp = line.split('\t')
        guid2entity[guid] = {'id': guid, 'word': word, 'type': tp}
    f.close()
    print("Finish loading, got {} entities totally".format(len(guid2entity)))


def get_instances(json_data, file_name):
    print("Loading {}".format(file_name))
    f = open(file_name, 'rb')
    buf = f.read()
    cur_pos = 0
    cnt_ins = 0
    cnt_rel_fact = 0
    while cur_pos < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, cur_pos)
        cur_pos = new_pos
        msg_buf = buf[cur_pos:cur_pos + msg_len]
        cur_pos += msg_len
        rel_fact = pb.Relation()
        cnt_rel_fact += 1
        rel_fact.ParseFromString(msg_buf)
        head = guid2entity[rel_fact.sourceGuid]
        tail = guid2entity[rel_fact.destGuid]
        relation = rel_fact.relType
        for ins in rel_fact.mention:
            cnt_ins += 1
            json_data.append({'sentence': ins.sentence, 'head': head, 'tail': tail, 'relation': relation})
    f.close()
    print("Finish loading, got {} instances and {} relfacts totally".format(cnt_ins, cnt_rel_fact))


def main():
    if len(sys.argv) != 2:
        raise Exception("[ERROR] You should specify the root directory for NYTData")
    root_dir = sys.argv[1]
    ori_train_json_data = []
    test_json_data = []
    get_entities(os.path.join(root_dir, 'filtered-freebase-simple-topic-dump-3cols.tsv'))
    get_instances(ori_train_json_data, os.path.join(root_dir, 'kb_manual/trainPositive.pb'))
    get_instances(ori_train_json_data, os.path.join(root_dir, 'kb_manual/trainNegative.pb'))
    get_instances(ori_train_json_data, os.path.join(root_dir, 'heldout_relations/trainPositive.pb'))
    get_instances(ori_train_json_data, os.path.join(root_dir, 'heldout_relations/trainNegative.pb'))
    get_instances(test_json_data, os.path.join(root_dir, 'heldout_relations/testPositive.pb'))
    get_instances(test_json_data, os.path.join(root_dir, 'heldout_relations/testNegative.pb'))

    print("Dealing with overlapping...")
    test_entities = {}
    for ins in test_json_data:
        test_entities[(ins['head']['word'] + '#' + ins['tail']['word']).lower()] = 1
    train_json_data = []
    for ins in ori_train_json_data:
        if not ((ins['head']['word'] + '#' + ins['tail']['word']).lower() in test_entities):
            train_json_data.append(ins)
    print("Finish")
    print("Storing json files...")
    json.dump(train_json_data, open('train.json', 'w'))
    json.dump(train_json_data, open('train-reading-friendly.json', 'w'), indent=4, sort_keys=True)
    json.dump(test_json_data, open('test.json', 'w'))
    json.dump(test_json_data, open('test-reading-friendly.json', 'w'), indent=4, sort_keys=True)
    print("Finish")


if __name__ == '__main__':
    main()
