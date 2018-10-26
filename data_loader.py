import json
import os
import random

import numpy as np
from six import iteritems


class file_data_loader:
    MODE_INSTANCE = 0  # One batch contains batch_size instances.
    MODE_ENTPAIR_BAG = 1  # One batch contains batch_size bags, instances in which have the same entity pair (usually for testing).
    MODE_RELFACT_BAG = 2  # One batch contains batch size bags, instances in which have the same relation fact. (usually for training).

    def __next__(self):
        raise NotImplementedError

    def next(self):
        return self.__next__()

    def next_batch(self, batch_size):
        raise NotImplementedError


class npy_data_loader(file_data_loader):
    def __iter__(self):
        return self

    def __init__(self, data_dir, prefix, mode, word_vec_npy='vec.npy', shuffle=True, max_length=120, batch_size=160):
        if not os.path.isdir(data_dir):
            raise Exception("[ERROR] Data dir doesn't exist!")
        self.mode = mode
        self.shuffle = shuffle
        self.max_length = max_length
        self.batch_size = batch_size
        self.word_vec = np.load(os.path.join(data_dir, word_vec_npy))
        self.data_word = np.load(os.path.join(data_dir, prefix + "_word.npy"))
        self.data_pos1 = np.load(os.path.join(data_dir, prefix + "_pos1.npy"))
        self.data_pos2 = np.load(os.path.join(data_dir, prefix + "_pos2.npy"))
        self.data_mask = np.load(os.path.join(data_dir, prefix + "_mask.npy"))
        self.data_length = np.load(os.path.join(data_dir, prefix + "_len.npy"))
        self.data_label = np.load(os.path.join(data_dir, prefix + "_label.npy"))
        self.scope = np.load(os.path.join(data_dir, prefix + "_instance_scope.npy"))
        self.triple = np.load(os.path.join(data_dir, prefix + "_instance_triple.npy"))
        self.relfact_tot = len(self.triple)
        for i in range(self.scope.shape[0]):
            self.scope[i][1] += 1

        self.instance_tot = self.data_word.shape[0]
        self.rel_tot = 53

        if self.mode == self.MODE_INSTANCE:
            self.order = list(range(self.instance_tot))
        else:
            self.order = list(range(len(self.scope)))
        self.idx = 0

        if self.shuffle:
            random.shuffle(self.order)

        print("Total relation fact: %d" % self.relfact_tot)

    def __next__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, batch_size):
        if self.idx >= len(self.order):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.order)
            raise StopIteration

        batch_data = {}

        if self.mode == self.MODE_INSTANCE:
            idx0 = self.idx
            idx1 = self.idx + batch_size
            if idx1 > len(self.order):
                self.idx = 0
                if self.shuffle:
                    random.shuffle(self.order)
                raise StopIteration
            self.idx = idx1
            batch_data['word'] = self.data_word[idx0:idx1]
            batch_data['pos1'] = self.data_pos1[idx0:idx1]
            batch_data['pos2'] = self.data_pos2[idx0:idx1]
            batch_data['mask'] = self.data_mask[idx0:idx1]
            batch_data['length'] = self.data_length[idx0:idx1]
            batch_data['label'] = self.data_label[idx0:idx1]
            # batch_data['scope'] = np.stack([list(range(idx1 - idx0)), list(range(1, idx1 - idx0 + 1))], axis=1)
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            idx0 = self.idx
            idx1 = self.idx + batch_size
            if idx1 > len(self.order):
                self.idx = 0
                if self.shuffle:
                    random.shuffle(self.order)
                raise StopIteration
            self.idx = idx1
            _word = []
            _pos1 = []
            _pos2 = []
            _mask = []
            _length = []
            _label = []
            _instance_label = []
            _scope = []
            _multi_label = []
            cur_pos = 0
            for i in range(idx0, idx1):
                _word.append(self.data_word[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos1.append(self.data_pos1[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos2.append(self.data_pos2[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _mask.append(self.data_mask[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _length.append(self.data_length[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _label.append(self.data_label[self.scope[self.order[i]][0]])
                _instance_label.append(self.data_label[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                bag_size = self.scope[self.order[i]][1] - self.scope[self.order[i]][0]
                _scope.append([cur_pos, cur_pos + bag_size])
                cur_pos = cur_pos + bag_size
                if self.mode == self.MODE_ENTPAIR_BAG:
                    _one_multi_rel = np.zeros(self.rel_tot, dtype=np.int32)
                    for j in range(self.scope[self.order[i]][0], self.scope[self.order[i]][1]):
                        _one_multi_rel[self.data_label[j]] = 1
                    _multi_label.append(_one_multi_rel)
            batch_data['word'] = np.concatenate(_word)
            batch_data['pos1'] = np.concatenate(_pos1)
            batch_data['pos2'] = np.concatenate(_pos2)
            batch_data['mask'] = np.concatenate(_mask)
            batch_data['length'] = np.concatenate(_length)
            batch_data['label'] = np.stack(_label)
            batch_data['instance_label'] = np.concatenate(_instance_label)
            batch_data['scope'] = np.stack(_scope)
            if self.mode == self.MODE_ENTPAIR_BAG:
                batch_data['multi_label'] = np.stack(_multi_label)

        return batch_data


class json_file_data_loader(file_data_loader):
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split(os.sep)[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split(os.sep)[-1].split('.')[:-1])
        processed_data_dir = 'processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_label.npy')
        entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        relfact2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_relfact2scope.json')
        word_vec_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
                not os.path.exists(pos1_npy_file_name) or \
                not os.path.exists(pos2_npy_file_name) or \
                not os.path.exists(label_npy_file_name) or \
                not os.path.exists(mask_npy_file_name) or \
                not os.path.exists(length_npy_file_name) or \
                not os.path.exists(entpair2scope_file_name) or \
                not os.path.exists(relfact2scope_file_name) or \
                not os.path.exists(word_vec_file_name) or \
                not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.data_label = np.load(label_npy_file_name)
        self.entpair2scope = json.load(open(entpair2scope_file_name))
        self.relfact2scope = json.load(open(relfact2scope_file_name))
        self.word_vec = np.load(word_vec_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, rel2id_file_name, mode, shuffle=True, max_length=120,
                 case_sensitive=False, reprocess=False, batch_size=160):
        """
        file_name: Json file storing the origin_data in the following format
            [
                {
                    'sentence': 'Bill Gates is the founder of Microsoft .',
                    'head': {'word': 'Bill Gates', ...(other information)},
                    'tail': {'word': 'Microsoft', ...(other information)},
                    'relation': 'founder'
                },
                ...
            ]
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        rel2id_file_name: Json file storing relation-to-id diction in the following format
            {
                'NA': 0
                'founder': 1
                ...
            }
            **IMPORTANT**: make sure the id of NA is 0!
        mode: Specify how to get a batch of origin_data. See MODE_* constants for details.
        shuffle: Whether to shuffle the origin_data, default as True. You should use shuffle when training.
        max_length: The length that all the sentences need to be extend to, default as 120.
        case_sensitive: Whether the origin_data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        batch_size: The size of each batch, default as 160.
        """

        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rel2id = json.load(open(rel2id_file_name))

        if reprocess or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading origin_data file...")
            self.origin_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.origin_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive
            if not case_sensitive:
                print("Eliminating case sensitive problem...")
                for i in range(len(self.origin_data)):
                    self.origin_data[i]['sentence'] = self.origin_data[i]['sentence'].lower()
                    self.origin_data[i]['head']['word'] = self.origin_data[i]['head']['word'].lower()
                    self.origin_data[i]['tail']['word'] = self.origin_data[i]['tail']['word'].lower()
                print("Finish eliminating")

            # Sort origin_data by entities and relations
            print("Sort origin_data...")
            self.origin_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
            print("Finish sorting")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.origin_word_vec)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.origin_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word_vec in enumerate(self.origin_word_vec):
                w = word_vec['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec[cur_id, :] = word_vec['vec']
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Pre-process origin_data
            print("Pre-processing origin_data...")
            self.instance_tot = len(self.origin_data)
            self.entpair2scope = {}  # (head, tail) -> scope
            self.relfact2scope = {}  # (head, tail, relation) -> scope
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros(self.instance_tot, dtype=np.int32)
            self.data_label = np.zeros(self.instance_tot, dtype=np.int32)
            last_entpair = ''
            last_entpair_pos = -1
            last_relfact = ''
            last_relfact_pos = -1
            for i in range(self.instance_tot):
                instance = self.origin_data[i]
                if instance['relation'] in self.rel2id:
                    self.data_label[i] = self.rel2id[instance['relation']]
                else:
                    self.data_label[i] = self.rel2id['NA']
                sentence = ' '.join(instance['sentence'].split())  # delete extra spaces
                head = instance['head']['word']
                tail = instance['tail']['word']
                cur_entpair = instance['head']['id'] + '#' + instance['tail']['id']
                cur_relfact = instance['head']['id'] + '#' + instance['tail']['id'] + '#' + instance['relation']
                if cur_entpair != last_entpair:
                    if last_entpair != '':
                        self.entpair2scope[last_entpair] = [last_entpair_pos, i]  # left closed right open
                    last_entpair = cur_entpair
                    last_entpair_pos = i
                if cur_relfact != last_relfact:
                    if last_relfact != '':
                        self.relfact2scope[last_relfact] = [last_relfact_pos, i]
                    last_relfact = cur_relfact
                    last_relfact_pos = i
                p1 = sentence.find(' ' + head + ' ')
                p2 = sentence.find(' ' + tail + ' ')
                if p1 == -1:
                    if sentence[:len(head) + 1] == head + " ":
                        p1 = 0
                    elif sentence[-len(head) - 1:] == " " + head:
                        p1 = len(sentence) - len(head)
                    else:
                        p1 = 0  # shouldn't happen
                else:
                    p1 += 1
                if p2 == -1:
                    if sentence[:len(tail) + 1] == tail + " ":
                        p2 = 0
                    elif sentence[-len(tail) - 1:] == " " + tail:
                        p2 = len(sentence) - len(tail)
                    else:
                        p2 = 0  # shouldn't happen
                else:
                    p2 += 1
                # if p1 == -1 or p2 == -1:
                #     raise Exception("[ERROR] Sentence doesn't contain the entity, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence, head, tail))

                words = sentence.split()
                cur_pos = 0
                pos1 = -1
                pos2 = -1
                for j, word in enumerate(words):
                    if j < max_length:
                        if word in self.word2id:
                            self.data_word[i][j] = self.word2id[word]
                        else:
                            self.data_word[i][j] = UNK
                    if cur_pos == p1:
                        pos1 = j
                        p1 = -1
                    if cur_pos == p2:
                        pos2 = j
                        p2 = -1
                    cur_pos += len(word) + 1
                for j in range(len(words), max_length):
                    self.data_word[i][j] = BLANK
                self.data_length[i] = len(words)
                if len(words) > max_length:
                    self.data_length[i] = max_length
                if pos1 == -1 or pos2 == -1:
                    raise Exception(
                        "[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence,
                                                                                                         head, tail))
                if pos1 >= max_length:
                    pos1 = max_length - 1
                if pos2 >= max_length:
                    pos2 = max_length - 1
                pos_min = min(pos1, pos2)
                pos_max = max(pos1, pos2)
                for j in range(max_length):
                    self.data_pos1[i][j] = j - pos1 + max_length
                    self.data_pos2[i][j] = j - pos2 + max_length
                    if j >= self.data_length[i]:
                        self.data_mask[i][j] = 0
                    elif j <= pos_min:
                        self.data_mask[i][j] = 1
                    elif j <= pos_max:
                        self.data_mask[i][j] = 2
                    else:
                        self.data_mask[i][j] = 3

            if last_entpair != '':
                self.entpair2scope[last_entpair] = [last_entpair_pos, self.instance_tot]  # left closed right open
            if last_relfact != '':
                self.relfact2scope[last_relfact] = [last_relfact_pos, self.instance_tot]

            print("Finish pre-processing")

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split(os.sep)[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split(os.sep)[-1].split('.')[:-1])
            processed_data_dir = 'processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_label.npy'), self.data_label)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.entpair2scope,
                      open(os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json'), 'w'))
            json.dump(self.relfact2scope,
                      open(os.path.join(processed_data_dir, name_prefix + '_relfact2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

        # Prepare for idx
        self.instance_tot = self.data_word.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = 0  # The number of relation facts, without NA.
        for key in self.relfact2scope:
            if key[-2:] != 'NA':
                self.relfact_tot += 1
        self.rel_tot = len(self.rel2id)

        if self.mode == self.MODE_INSTANCE:
            self.order = list(range(self.instance_tot))
        elif self.mode == self.MODE_ENTPAIR_BAG:
            self.order = list(range(len(self.entpair2scope)))
            self.scope_name = []
            self.scope = []
            for key, value in iteritems(self.entpair2scope):
                self.scope_name.append(key)
                self.scope.append(value)
        elif self.mode == self.MODE_RELFACT_BAG:
            self.order = list(range(len(self.relfact2scope)))
            self.scope_name = []
            self.scope = []
            for key, value in iteritems(self.relfact2scope):
                self.scope_name.append(key)
                self.scope.append(value)
        else:
            raise Exception("[ERROR] Invalid mode")
        self.idx = 0

        if self.shuffle:
            random.shuffle(self.order)

        print("Total relation fact: %d" % self.relfact_tot)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, batch_size):
        if self.idx >= len(self.order):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.order)
            raise StopIteration

        batch_data = {}

        if self.mode == self.MODE_INSTANCE:
            idx0 = self.idx
            idx1 = self.idx + batch_size
            if idx1 > len(self.order):
                idx1 = len(self.order)
            self.idx = idx1
            batch_data['word'] = self.data_word[idx0:idx1]
            batch_data['pos1'] = self.data_pos1[idx0:idx1]
            batch_data['pos2'] = self.data_pos2[idx0:idx1]
            batch_data['mask'] = self.data_mask[idx0:idx1]
            batch_data['length'] = self.data_length[idx0:idx1]
            batch_data['label'] = self.data_label[idx0:idx1]
            # batch_data['scope'] = np.stack([list(range(batch_size)), list(range(1, batch_size + 1))], axis=1)
            if idx1 - idx0 < batch_size:
                padding = batch_size - (idx1 - idx0)
                batch_data['word'] = np.concatenate(
                    [batch_data['word'], np.zeros((padding, self.data_word.shape[-1]), dtype=np.int32)])
                batch_data['pos1'] = np.concatenate(
                    [batch_data['pos1'], np.zeros((padding, self.data_pos1.shape[-1]), dtype=np.int32)])
                batch_data['pos2'] = np.concatenate(
                    [batch_data['pos2'], np.zeros((padding, self.data_pos2.shape[-1]), dtype=np.int32)])
                batch_data['mask'] = np.concatenate(
                    [batch_data['mask'], np.zeros((padding, self.data_mask.shape[-1]), dtype=np.int32)])
                batch_data['label'] = np.concatenate([batch_data['label'], np.zeros(padding, dtype=np.int32)])
                batch_data['length'] = np.concatenate([batch_data['length'], np.zeros(padding, dtype=np.int32)])
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            idx0 = self.idx
            idx1 = self.idx + batch_size
            if idx1 > len(self.order):
                idx1 = len(self.order)
            self.idx = idx1
            _word = []
            _pos1 = []
            _pos2 = []
            _mask = []
            _length = []
            _label = []
            _instance_label = []
            _multi_label = []
            _entpair = []
            _scope = []
            cur_pos = 0
            for i in range(idx0, idx1):
                _word.append(self.data_word[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos1.append(self.data_pos1[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _pos2.append(self.data_pos2[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _mask.append(self.data_mask[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _length.append(self.data_length[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                _label.append(self.data_label[self.scope[self.order[i]][0]])
                _instance_label.append(self.data_label[self.scope[self.order[i]][0]:self.scope[self.order[i]][1]])
                bag_size = self.scope[self.order[i]][1] - self.scope[self.order[i]][0]
                _scope.append([cur_pos, cur_pos + bag_size])
                cur_pos = cur_pos + bag_size
                if self.mode == self.MODE_ENTPAIR_BAG:
                    _one_multi_rel = np.zeros(self.rel_tot, dtype=np.int32)
                    for j in range(self.scope[self.order[i]][0], self.scope[self.order[i]][1]):
                        _one_multi_rel[self.data_label[j]] = 1
                    _multi_label.append(_one_multi_rel)
                    _entpair.append(self.scope_name[self.order[i]])
            for i in range(batch_size - (idx1 - idx0)):
                _word.append(np.zeros((1, self.data_word.shape[-1]), dtype=np.int32))
                _pos1.append(np.zeros((1, self.data_pos1.shape[-1]), dtype=np.int32))
                _pos2.append(np.zeros((1, self.data_pos2.shape[-1]), dtype=np.int32))
                _mask.append(np.zeros((1, self.data_mask.shape[-1]), dtype=np.int32))
                _length.append(np.zeros(1, dtype=np.int32))
                _label.append(0)
                _instance_label.append(np.zeros(1, dtype=np.int32))
                _scope.append([cur_pos, cur_pos + 1])
                cur_pos += 1
                if self.mode == self.MODE_ENTPAIR_BAG:
                    _multi_label.append(np.zeros(self.rel_tot, dtype=np.int32))
                    _entpair.append('None#None')
            batch_data['word'] = np.concatenate(_word)
            batch_data['pos1'] = np.concatenate(_pos1)
            batch_data['pos2'] = np.concatenate(_pos2)
            batch_data['mask'] = np.concatenate(_mask)
            batch_data['length'] = np.concatenate(_length)
            batch_data['label'] = np.stack(_label)
            batch_data['instance_label'] = np.concatenate(_instance_label)
            if self.mode == self.MODE_ENTPAIR_BAG:
                batch_data['multi_label'] = np.stack(_multi_label)
                batch_data['entpair'] = _entpair
            batch_data['scope'] = np.stack(_scope)

        return batch_data
