import json
import os
import random

import numpy as np
import tensorflow as tf
from six import iteritems

FLAGS = tf.flags.FLAGS


class file_data_loader:
    MODE_INSTANCE = 0  # One batch contains batch_size instances.
    MODE_ENTPAIR_BAG = 1  # One batch contains batch_size bags, instances in which have the same entity pair (usually for testing).
    MODE_RELFACT_BAG = 2  # One batch contains batch size bags, instances in which have the same relation fact. (usually for training).
    TRAIN_PREFIX = 'train'
    TEST_PREFIX = 'test'

    def __init__(self, ext, prefix=TRAIN_PREFIX, mode=MODE_RELFACT_BAG, reprocess=False, shuffle=True):
        self.ext = ext
        self.prefix = prefix
        self.mode = mode
        self.shuffle = shuffle

        if reprocess or not self._load_preprocessed_file():
            self._preprocess()

        self.instance_tot = self.data_word.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = 0
        self.rel_tot = len(self.rel2id)

        for k in self.relfact2scope:
            if k[-2:] != 'NA':
                self.relfact_tot += 1

        self.order = []
        self.scope_name = []
        self.scope = []
        self.set_order()
        self.begin = 0

        print("Total relation fact: %d" % self.relfact_tot)
        if self.prefix == self.TRAIN_PREFIX:
            self._set_weights_table()

    def set_order(self):
        if self.mode == self.MODE_INSTANCE:
            self.order = list(range(self.instance_tot))
        else:
            order_item = None
            if self.mode == self.MODE_ENTPAIR_BAG:
                order_item = self.entpair2scope
            elif self.mode == self.MODE_RELFACT_BAG:
                order_item = self.relfact2scope
            if order_item is None:
                raise Exception("[ERROR] Invalid mode")
            self.order = list(range(len(order_item)))
            self.scope_name = []
            self.scope = []
            for k, v in iteritems(order_item):
                self.scope_name.append(k)
                self.scope.append(v)
            self.scope = np.array(self.scope)
        if self.shuffle:
            random.shuffle(self.order)

    def _load_preprocessed_file(self):
        if not os.path.isdir(FLAGS.processed_data_dir):
            return False
        word_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_word.npy')
        pos1_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_pos1.npy')
        pos2_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_pos2.npy')
        mask_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_mask.npy')
        length_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_length.npy')
        label_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_label.npy')
        word_vec_file_name = os.path.join(FLAGS.processed_data_dir, 'word_vec.npy')
        entpair2scope_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_entpair2scope' + self.ext)
        relfact2scope_file_name = os.path.join(FLAGS.processed_data_dir, self.prefix + '_relfact2scope' + self.ext)
        rel2id_file_name = os.path.join(FLAGS.processed_data_dir, 'rel2id' + self.ext)
        word2id_file_name = os.path.join(FLAGS.processed_data_dir, 'word2id' + self.ext)
        if not os.path.exists(word_file_name) or not os.path.exists(pos1_file_name) or \
                not os.path.exists(pos2_file_name) or not os.path.exists(mask_file_name) or \
                not os.path.exists(length_file_name) or not os.path.exists(label_file_name) or \
                not os.path.exists(word_vec_file_name) or \
                not os.path.exists(entpair2scope_file_name) or not os.path.exists(relfact2scope_file_name) or \
                not os.path.exists(rel2id_file_name) or not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_file_name)
        self.data_pos1 = np.load(pos1_file_name)
        self.data_pos2 = np.load(pos2_file_name)
        self.data_mask = np.load(mask_file_name)
        self.data_length = np.load(length_file_name)
        self.data_label = np.load(label_file_name)
        self.word_vec = np.load(word_vec_file_name)
        self.entpair2scope = self.load_file(entpair2scope_file_name)
        self.relfact2scope = self.load_file(relfact2scope_file_name)
        self.rel2id = self.load_file(rel2id_file_name)
        self.word2id = self.load_file(word2id_file_name)
        if self.data_word.shape[1] != FLAGS.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def _preprocess(self, case_sensitive=False):
        origin_data_file_name = os.path.join(FLAGS.dataset_dir, self.prefix + self.ext)
        word_vec_file_name = os.path.join(FLAGS.dataset_dir, 'word_vec' + self.ext)
        rel2id_file_name = os.path.join(FLAGS.dataset_dir, 'rel2id' + self.ext)
        # Check files
        if origin_data_file_name is None or not os.path.isfile(origin_data_file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
            raise Exception("[ERROR] Word vector file doesn't exist")

        # Load files
        print("Loading origin_data file...")
        self.origin_data = self.load_file(origin_data_file_name)
        print("Finish origin_data loading")
        print("Loading word vector file...")
        self.origin_word_vec = self.load_file(word_vec_file_name)
        print("Finish word vector loading")
        print("Loading rel2id file...")
        self.rel2id = self.load_file(rel2id_file_name)
        print("Finish rel2id loading")

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
        for i, word_vec in enumerate(self.origin_word_vec):
            w = word_vec['word']
            if not case_sensitive:
                w = w.lower()
            self.word2id[w] = i
            self.word_vec[i, :] = word_vec['vec']
        self.word2id['UNK'] = UNK
        self.word2id['BLANK'] = BLANK
        print("Finish building")

        # Pre-process origin_data
        print("Pre-processing origin_data...")
        self.instance_tot = len(self.origin_data)
        self.data_word = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_pos1 = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_pos2 = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_mask = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_length = np.zeros(self.instance_tot, dtype=np.int32)
        self.data_label = np.zeros(self.instance_tot, dtype=np.int32)
        self.entpair2scope = {}  # (head, tail) -> scope
        self.relfact2scope = {}  # (head, tail, relation) -> scope
        last_entpair = ''
        last_entpair_pos = -1
        last_relfact = ''
        last_relfact_pos = -1
        for i in range(self.instance_tot):
            instance = self.origin_data[i]
            sentence = ' '.join(instance['sentence'].split())  # delete extra spaces
            head = instance['head']['word']
            tail = instance['tail']['word']

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
            #     raise Exception("[ERROR] Sentence doesn't contain the entity, index = {}, sentence = {}, head = {},
            #                     tail = {}".format(i, sentence, head, tail))

            words = sentence.split()
            cur_pos = 0
            pos1 = -1
            pos2 = -1
            for j, word in enumerate(words):
                if j < FLAGS.max_length:
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
            for j in range(len(words), FLAGS.max_length):
                self.data_word[i][j] = BLANK

            if pos1 == -1 or pos2 == -1:
                raise Exception(
                    "[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence,
                                                                                                     head, tail))
            if pos1 >= FLAGS.max_length:
                pos1 = FLAGS.max_length - 1
            if pos2 >= FLAGS.max_length:
                pos2 = FLAGS.max_length - 1
            pos_min = min(pos1, pos2)
            pos_max = max(pos1, pos2)
            for j in range(FLAGS.max_length):
                self.data_pos1[i][j] = j - pos1 + FLAGS.max_length
                self.data_pos2[i][j] = j - pos2 + FLAGS.max_length
                if j >= self.data_length[i]:
                    self.data_mask[i][j] = 0
                elif j <= pos_min:
                    self.data_mask[i][j] = 1
                elif j <= pos_max:
                    self.data_mask[i][j] = 2
                else:
                    self.data_mask[i][j] = 3

            self.data_length[i] = len(words)
            if len(words) > FLAGS.max_length:
                self.data_length[i] = FLAGS.max_length

            if instance['relation'] in self.rel2id:
                self.data_label[i] = self.rel2id[instance['relation']]
            else:
                self.data_label[i] = self.rel2id['NA']

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
        if last_entpair != '':
            self.entpair2scope[last_entpair] = [last_entpair_pos, self.instance_tot]  # left closed right open
        if last_relfact != '':
            self.relfact2scope[last_relfact] = [last_relfact_pos, self.instance_tot]

        print("Finish pre-processing")

        self._store_processed_files()

    @staticmethod
    def load_file(file_name):
        raise NotImplementedError

    @staticmethod
    def save_file(data, file_name):
        raise NotImplementedError

    def _store_processed_files(self):
        print("Storing processed files...")
        if not os.path.isdir(FLAGS.processed_data_dir):
            os.mkdir(FLAGS.processed_data_dir)
        np.save(os.path.join(FLAGS.processed_data_dir, self.prefix + '_word.npy'), self.data_word)
        np.save(os.path.join(FLAGS.processed_data_dir, self.prefix + '_pos1.npy'), self.data_pos1)
        np.save(os.path.join(FLAGS.processed_data_dir, self.prefix + '_pos2.npy'), self.data_pos2)
        np.save(os.path.join(FLAGS.processed_data_dir, self.prefix + '_mask.npy'), self.data_mask)
        np.save(os.path.join(FLAGS.processed_data_dir, self.prefix + '_length.npy'), self.data_length)
        np.save(os.path.join(FLAGS.processed_data_dir, self.prefix + '_label.npy'), self.data_label)
        np.save(os.path.join(FLAGS.processed_data_dir, 'word_vec.npy'), self.word_vec)
        self.save_file(self.entpair2scope, self.prefix + '_entpair2scope' + self.ext)
        self.save_file(self.relfact2scope, self.prefix + '_relfact2scope' + self.ext)
        self.save_file(self.rel2id, 'rel2id' + self.ext)
        self.save_file(self.word2id, 'word2id' + self.ext)
        print("Finish storing")

    def _set_weights_table(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            self.weights_table = np.zeros(self.rel_tot, dtype=np.float32)
            for i in range(len(self.data_label)):
                self.weights_table[self.data_label[i]] += 1.0
            self.weights_table = [weight if weight == 0 else 1 / (weight ** 0.05) for weight in self.weights_table]

            print("Finish calculating")

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(FLAGS.batch_size)

    def batch_data(self, index, scope=None, multi_label=None):
        batch_data = {'word': self.data_word[index], 'pos1': self.data_pos1[index], 'pos2': self.data_pos2[index],
                      'mask': self.data_mask[index], 'length': self.data_length[index]}
        if self.mode == self.MODE_INSTANCE:
            batch_data.update({'label': self.data_label[index]})
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            batch_data.update({'label': self.data_label[index][scope[:-1]]})
            batch_data.update({'instance_label': self.data_label[index]})
        if hasattr(self, 'scope') and scope is not None:
            batch_data.update({'scope': scope})
        if hasattr(self, 'weights_table'):
            batch_data.update({'weights': [self.weights_table[label] for label in batch_data['label']]})
        if multi_label is not None:
            batch_data.update({'multi_label': multi_label})
        return batch_data

    def add2batch_data(self, batch_data, b, e):
        batch_data['word'] = np.concatenate([batch_data['word'], self.data_word[b:e]]) \
            if batch_data.__contains__('word') else self.data_word[b:e]
        batch_data['pos1'] = np.concatenate([batch_data['pos1'], self.data_pos1[b:e]]) \
            if batch_data.__contains__('pos1') else self.data_pos1[b:e]
        batch_data['pos2'] = np.concatenate([batch_data['pos2'], self.data_pos2[b:e]]) \
            if batch_data.__contains__('pos2') else self.data_pos2[b:e]
        batch_data['mask'] = np.concatenate([batch_data['mask'], self.data_mask[b:e]]) \
            if batch_data.__contains__('mask') else self.data_mask[b:e]
        batch_data['length'] = np.concatenate([batch_data['length'], self.data_length[b:e]]) \
            if batch_data.__contains__('length') else self.data_length[b:e]

    def next_batch(self, batch_size):
        if self.begin >= len(self.order):
            self.begin = 0
            if self.shuffle:
                random.shuffle(self.order)
            raise StopIteration

        i = self.begin
        j = self.begin + batch_size
        if i > len(self.order):
            self.begin = 0
            if self.shuffle:
                random.shuffle(self.order)
            raise StopIteration
        self.begin = j

        batch_data = {}
        if self.mode == self.MODE_INSTANCE:
            self.add2batch_data(batch_data, i, j)
            batch_data['label'] = self.data_label[i:j]
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            scope = np.zeros(batch_size + 1, dtype=np.int32)
            multi_label = [] if self.mode == self.MODE_ENTPAIR_BAG else None
            for p, k in enumerate(range(i, j)):
                b, e = self.scope[self.order[k]][0], self.scope[self.order[k]][1]
                scope[p + 1] = scope[p] + e - b
                if multi_label is not None:
                    _one_multi_rel = np.zeros(self.rel_tot, dtype=np.int32)
                    for n in range(self.scope[self.order[k]][0], self.scope[self.order[k]][1]):
                        _one_multi_rel[self.data_label[n]] = 1
                    multi_label.append(_one_multi_rel)
                    batch_data['multi_label'] = np.concatenate([batch_data['multi_label'], multi_label]) \
                        if batch_data.__contains__('multi_label') else multi_label

                self.add2batch_data(batch_data, b, e)
                batch_data['label'] = np.concatenate([batch_data['label'], [self.data_label[b]]]) \
                    if batch_data.__contains__('label') else [self.data_label[b]]
                batch_data['instance_label'] = np.concatenate([batch_data['instance_label'], self.data_label[b:e]]) \
                    if batch_data.__contains__('instance_label') else self.data_label[b:e]

            batch_data['scope'] = scope

        if hasattr(self, 'weights_table'):
            batch_data.update({'weights': [self.weights_table[label] for label in batch_data['label']]})

        return batch_data


class npy_data_loader(file_data_loader):
    def __init__(self, prefix=file_data_loader.TRAIN_PREFIX, mode=file_data_loader.MODE_RELFACT_BAG,
                 reprocess=False, shuffle=True):
        super().__init__('.npy', prefix, mode, reprocess, shuffle)

    @staticmethod
    def load_file(file_name):
        return np.load(file_name)

    @staticmethod
    def save_file(data, file_name):
        np.save(os.path.join(FLAGS.processed_data_dir, file_name), data)


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


class json_file_data_loader(file_data_loader):
    def __init__(self, prefix=file_data_loader.TRAIN_PREFIX, mode=file_data_loader.MODE_RELFACT_BAG,
                 reprocess=False, shuffle=True):
        super().__init__('.json', prefix, mode, reprocess, shuffle)

    @staticmethod
    def load_file(file_name):
        return json.load(open(file_name, "r"))

    @staticmethod
    def save_file(data, file_name):
        json.dump(data, open(os.path.join(FLAGS.processed_data_dir, file_name), 'w'))
