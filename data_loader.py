import json
import os
import random

import numpy as np
import tensorflow as tf
from six import iteritems

import util

FLAGS = tf.flags.FLAGS


class file_data_loader:
    MODE_INSTANCE = 0  # One batch contains batch_size instances.
    MODE_ENTPAIR_BAG = 1  # One batch contains batch_size bags, instances in which have the same entity pair (usually for testing).
    MODE_RELFACT_BAG = 2  # One batch contains batch size bags, instances in which have the same relation fact. (usually for training).
    TRAIN_PREFIX = 'train'
    TEST_PREFIX = 'test'
    case_sensitive = False
    word_vec_tot = 0
    word_vec_dim = 0
    UNK = 0
    BLANK = 0
    origin_word_vec = None
    word_vec = None
    word2id = {}
    rel2id = {}
    enttype2id = {}
    rel_tot = 0
    enttype_tot = 0

    def __init__(self, ext, prefix=TRAIN_PREFIX, mode=MODE_RELFACT_BAG, reprocess=False, shuffle=True):
        self.ext = ext
        self.prefix = prefix
        self.mode = mode
        self.shuffle = shuffle
        self.data_require = {}

        if reprocess or not self._load_preprocessed_file():
            self._preprocess()

        self.instance_tot = self.data_word.shape[0]
        self.entpair_tot = len(self.entpair2scope)
        self.relfact_tot = 0
        file_data_loader.rel_tot = len(file_data_loader.rel2id)
        file_data_loader.enttype_tot = len(file_data_loader.enttype2id)

        for k in self.relfact2scope:
            if k[-2:] != 'NA':
                self.relfact_tot += 1

        self.order = []
        self.scope_name = []
        self.scope = []
        self.set_order()

        self.begin = 0
        print("Total relation fact: %d" % self.relfact_tot)
        if self.prefix == file_data_loader.TRAIN_PREFIX:
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
            for k, v in iteritems(order_item):
                self.scope_name.append(k)
                self.scope.append(v)
        if self.shuffle:
            random.shuffle(self.order)

    def _load_preprocessed_file(self):
        if not os.path.exists(FLAGS.processed_data_dir):
            os.makedirs(FLAGS.processed_data_dir)
        word = os.path.join(FLAGS.processed_data_dir, self.prefix + '_word.npy')
        pos1 = os.path.join(FLAGS.processed_data_dir, self.prefix + '_pos1.npy')
        pos2 = os.path.join(FLAGS.processed_data_dir, self.prefix + '_pos2.npy')
        mask = os.path.join(FLAGS.processed_data_dir, self.prefix + '_mask.npy')
        length = os.path.join(FLAGS.processed_data_dir, self.prefix + '_length.npy')
        label = os.path.join(FLAGS.processed_data_dir, self.prefix + '_label.npy')
        entity_pos = os.path.join(FLAGS.processed_data_dir, self.prefix + '_entity_pos.npy')
        head_enttype = os.path.join(FLAGS.processed_data_dir, self.prefix + '_head_enttype.npy')
        tail_enttype = os.path.join(FLAGS.processed_data_dir, self.prefix + '_tail_enttype.npy')
        enttype_length = os.path.join(FLAGS.processed_data_dir, self.prefix + '_enttype_length.npy')
        word_vec = os.path.join(FLAGS.processed_data_dir, 'word_vec.npy')
        entpair2scope = os.path.join(FLAGS.processed_data_dir, self.prefix + '_entpair2scope' + self.ext)
        relfact2scope = os.path.join(FLAGS.processed_data_dir, self.prefix + '_relfact2scope' + self.ext)
        word2id = os.path.join(FLAGS.processed_data_dir, 'word2id' + self.ext)
        rel2id = os.path.join(FLAGS.processed_data_dir, 'rel2id' + self.ext)
        enttype2id = os.path.join(FLAGS.processed_data_dir, 'enttype2id' + self.ext)

        self.f_exists = {'word': os.path.exists(word), 'pos1': os.path.exists(pos1), 'pos2': os.path.exists(pos2),
                         'mask': os.path.exists(mask), 'length': os.path.exists(length), 'label': os.path.exists(label),
                         'head_enttype': os.path.exists(head_enttype), 'tail_enttype': os.path.exists(tail_enttype),
                         'enttype_length': os.path.exists(enttype_length), 'word_vec': os.path.exists(word_vec),
                         'entpair2scope': os.path.exists(entpair2scope), 'relfact2scope': os.path.exists(relfact2scope),
                         'word2id': os.path.exists(word2id), 'rel2id': os.path.exists(rel2id),
                         'enttype2id': os.path.exists(enttype2id), 'entity_pos': os.path.exists(entity_pos)}
        if False in self.f_exists.values():
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word)
        self.data_pos1 = np.load(pos1)
        self.data_pos2 = np.load(pos2)
        # self.data_mask = np.load(mask)
        self.data_length = np.load(length)
        self.data_label = np.load(label)
        self.entity_pos = np.load(entity_pos)
        self.head_enttype = np.load(head_enttype)
        self.tail_enttype = np.load(tail_enttype)
        self.enttype_length = np.load(enttype_length)
        if file_data_loader.word_vec is None:
            file_data_loader.word_vec = np.load(word_vec)
        self.entpair2scope = self.load_file(entpair2scope)
        self.relfact2scope = self.load_file(relfact2scope)
        # if len(file_data_loader.word2id) == 0:
        #     file_data_loader.word2id = self.load_file(word2id)
        if len(file_data_loader.rel2id) == 0:
            file_data_loader.rel2id = self.load_file(rel2id)
        if len(file_data_loader.enttype2id) == 0:
            file_data_loader.enttype2id = self.load_file(enttype2id)
        if self.data_word.shape[1] != FLAGS.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def _preprocess(self):
        # load origin files
        self._load_origin_files()

        # eliminate case sensitive
        self._eliminate_case_sensitive()

        # sort origin_data by entities and relations
        print("Sort origin_data...")
        self.origin_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
        print("Finish sorting")

        # preprocess origin_data
        self._preprocess_origin_data()

        self._store_processed_files()

    def _load_origin_files(self):
        origin_data = os.path.join(FLAGS.dataset_dir, self.prefix + self.ext)
        origin_word_vec = os.path.join(FLAGS.dataset_dir, 'word_vec' + self.ext)
        origin_rel2id = os.path.join(FLAGS.dataset_dir, 'rel2id' + self.ext)
        origin_enttype2id = os.path.join(FLAGS.dataset_dir, 'enttype2id' + self.ext)

        if origin_data is None or not os.path.isfile(origin_data):
            raise Exception("[ERROR] data file doesn't exist")
        print("Loading origin_data file...")
        self.origin_data = self.load_file(origin_data)
        print("Finish origin_data loading")
        if not self.f_exists['word_vec'] or not self.f_exists['word2id']:
            if origin_word_vec is None or not os.path.isfile(origin_word_vec):
                raise Exception("[ERROR] word vector file doesn't exist")
            print("Loading word vector file...")
            file_data_loader.origin_word_vec = self.load_file(origin_word_vec)
            print("Finish word vector loading")
            # preprocess word vec
            self._preprocess_word_vec()
        if file_data_loader.word_vec is None or len(file_data_loader.word2id) == 0:
            file_data_loader.word_vec = np.load(os.path.join(FLAGS.processed_data_dir, 'word_vec.npy'))
            file_data_loader.word2id = self.load_file(os.path.join(FLAGS.processed_data_dir, 'word2id' + self.ext))
            file_data_loader.word_vec_tot = len(file_data_loader.word_vec)
            file_data_loader.word_vec_dim = len(file_data_loader.word_vec[0])
            file_data_loader.UNK = file_data_loader.word_vec_tot
            file_data_loader.BLANK = file_data_loader.word_vec_tot + 1
        if not self.f_exists['rel2id'] or len(file_data_loader.rel2id) == 0:
            if origin_rel2id is None or not os.path.isfile(origin_rel2id):
                raise Exception("[ERROR] rel2id file doesn't exist")
            print("Loading rel2id file...")
            file_data_loader.rel2id = self.load_file(origin_rel2id)
            print("Finish rel2id loading")
        if not self.f_exists['enttype2id'] or len(file_data_loader.enttype2id) == 0:
            if origin_enttype2id is None or not os.path.isfile(origin_enttype2id):
                raise Exception("[ERROR] enttype2id file doesn't exist")
            print("Loading enttype2id file...")
            file_data_loader.enttype2id = self.load_file(origin_enttype2id)
            print("Finish enttype2id loading")

    def _eliminate_case_sensitive(self):
        if not file_data_loader.case_sensitive:
            print("Eliminating case sensitive problem...")
            for i in range(len(self.origin_data)):
                self.origin_data[i]['sentence'] = self.origin_data[i]['sentence'].lower()
                self.origin_data[i]['head']['word'] = self.origin_data[i]['head']['word'].lower()
                self.origin_data[i]['tail']['word'] = self.origin_data[i]['tail']['word'].lower()
            print("Finish eliminating")

    @staticmethod
    def _preprocess_word_vec():
        file_data_loader.word_vec_tot = len(file_data_loader.origin_word_vec)
        file_data_loader.word_vec_dim = len(file_data_loader.origin_word_vec[0]['vec'])
        file_data_loader.UNK = file_data_loader.word_vec_tot
        file_data_loader.BLANK = file_data_loader.word_vec_tot + 1
        print("Got {} words of {} dims".format(file_data_loader.word_vec_tot, file_data_loader.word_vec_dim))
        print("Building word vector matrix and mapping...")
        file_data_loader.word_vec = np.zeros((file_data_loader.word_vec_tot, file_data_loader.word_vec_dim),
                                             dtype=np.float32)
        for i, word_vec in enumerate(file_data_loader.origin_word_vec):
            w = word_vec['word']
            if not file_data_loader.case_sensitive:
                w = w.lower()
            file_data_loader.word2id[w] = i
            file_data_loader.word_vec[i, :] = word_vec['vec']
        file_data_loader.word2id['UNK'] = file_data_loader.UNK
        file_data_loader.word2id['BLANK'] = file_data_loader.BLANK
        print("Finish building")

    def _preprocess_origin_data(self):
        print("Pre-processing origin_data...")
        self.instance_tot = len(self.origin_data)
        self.data_word = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_pos1 = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_pos2 = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_mask = np.zeros((self.instance_tot, FLAGS.max_length), dtype=np.int32)
        self.data_length = np.zeros(self.instance_tot, dtype=np.int32)
        self.data_label = np.zeros(self.instance_tot, dtype=np.int32)
        self.entity_pos = np.zeros((self.instance_tot, 2), dtype=np.int32)
        self.head_enttype = np.zeros((self.instance_tot, FLAGS.et_max_length), dtype=np.int32)
        self.tail_enttype = np.zeros((self.instance_tot, FLAGS.et_max_length), dtype=np.int32)
        self.enttype_length = np.zeros((self.instance_tot, 2), dtype=np.int32)
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
            self.data_length[i] = len(words)
            if len(words) > FLAGS.max_length:
                self.data_length[i] = FLAGS.max_length

            cur_pos = 0
            pos1 = -1
            pos2 = -1
            for j, word in enumerate(words):
                if j < FLAGS.max_length:
                    if word in self.word2id:
                        self.data_word[i][j] = file_data_loader.word2id[word]
                    else:
                        self.data_word[i][j] = file_data_loader.UNK
                if cur_pos == p1:
                    pos1 = j
                    p1 = -1
                if cur_pos == p2:
                    pos2 = j
                    p2 = -1
                cur_pos += len(word) + 1
            for j in range(len(words), FLAGS.max_length):
                self.data_word[i][j] = file_data_loader.BLANK

            if pos1 == -1 or pos2 == -1:
                raise Exception(
                    "[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sentence,
                                                                                                     head, tail))
            if pos1 >= FLAGS.max_length:
                pos1 = FLAGS.max_length - 1
            if pos2 >= FLAGS.max_length:
                pos2 = FLAGS.max_length - 1
            self.entity_pos[i][0] = min(pos1, pos2)
            self.entity_pos[i][1] = max(pos1, pos2)
            for j in range(FLAGS.max_length):
                self.data_pos1[i][j] = j - pos1 + FLAGS.max_length
                self.data_pos2[i][j] = j - pos2 + FLAGS.max_length
                if j >= self.data_length[i]:
                    self.data_mask[i][j] = 0
                elif j <= self.entity_pos[i][0]:
                    self.data_mask[i][j] = 1
                elif j <= self.entity_pos[i][1]:
                    self.data_mask[i][j] = 2
                else:
                    self.data_mask[i][j] = 3

            if instance['relation'] in file_data_loader.rel2id:
                self.data_label[i] = file_data_loader.rel2id[instance['relation']]
            else:
                self.data_label[i] = file_data_loader.rel2id['NA']

            types = instance['head']['type'].split(',')
            self.enttype_length[i][0] = len(types)
            for j, t in enumerate(types):
                if t in file_data_loader.enttype2id:
                    self.head_enttype[i][j] = file_data_loader.enttype2id[t]
                else:
                    self.head_enttype[i][j] = len(file_data_loader.enttype2id)
            for j in range(len(types), FLAGS.et_max_length):
                self.head_enttype[i][j] = len(file_data_loader.enttype2id) + 1
            types = instance['tail']['type'].split(',')
            self.enttype_length[i][1] = len(types)
            for j, t in enumerate(types):
                if t in file_data_loader.enttype2id:
                    self.tail_enttype[i][j] = file_data_loader.enttype2id[t]
                else:
                    self.tail_enttype[i][j] = len(file_data_loader.enttype2id)
            for j in range(len(types), FLAGS.et_max_length):
                self.tail_enttype[i][j] = len(file_data_loader.enttype2id) + 1

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

    def np_save_file(self, data, file_name):
        util.con_exec(not self.f_exists[file_name[len(self.prefix) + 1 if file_name.startswith(self.prefix) else 0:-4]],
                      lambda: np.save(os.path.join(FLAGS.processed_data_dir, file_name), data))

    def json_save_file(self, data, file_name):
        util.con_exec(not self.f_exists[file_name[len(self.prefix) + 1 if file_name.startswith(self.prefix) else 0:-5]],
                      lambda: json.dump(data, open(os.path.join(FLAGS.processed_data_dir, file_name), 'w')))

    @staticmethod
    def load_file(file_name):
        raise NotImplementedError

    def save_file(self, data, file_name):
        raise NotImplementedError

    def _store_processed_files(self):
        print("Storing processed files...")
        self.np_save_file(self.data_word, self.prefix + '_word.npy')
        self.np_save_file(self.data_pos1, self.prefix + '_pos1.npy')
        self.np_save_file(self.data_pos2, self.prefix + '_pos2.npy')
        self.np_save_file(self.data_mask, self.prefix + '_mask.npy')
        self.np_save_file(self.data_length, self.prefix + '_length.npy')
        self.np_save_file(self.data_label, self.prefix + '_label.npy')
        self.np_save_file(self.entity_pos, self.prefix + '_entity_pos.npy')
        self.np_save_file(self.head_enttype, self.prefix + '_head_enttype.npy')
        self.np_save_file(self.tail_enttype, self.prefix + '_tail_enttype.npy')
        self.np_save_file(self.enttype_length, self.prefix + '_enttype_length.npy')
        self.np_save_file(file_data_loader.word_vec, 'word_vec.npy')
        self.save_file(self.entpair2scope, self.prefix + '_entpair2scope' + self.ext)
        self.save_file(self.relfact2scope, self.prefix + '_relfact2scope' + self.ext)
        self.save_file(file_data_loader.word2id, 'word2id' + self.ext)
        self.save_file(file_data_loader.rel2id, 'rel2id' + self.ext)
        self.save_file(file_data_loader.enttype2id, 'enttype2id' + self.ext)
        del self.data_mask, self.word2id
        print("Finish storing")

    def _set_weights_table(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            self.weights_table = np.zeros(self.rel_tot, dtype=np.float32)
            for i in range(len(self.data_label)):
                self.weights_table[self.data_label[i]] += 1.0
            self.weights_table = [weight if weight == 0 else 1 / (weight ** 0.05) for weight in self.weights_table]

            print("Finish calculating")

    def get_weights(self, labels):
        return [self.weights_table[label] for label in labels]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(FLAGS.batch_size)

    def batch_data(self, index, scope=None, multi_label=None):
        batch_data = {'word': self.data_word[index], 'pos1': self.data_pos1[index], 'pos2': self.data_pos2[index],
                      'mask': self.data_mask[index], 'length': self.data_length[index],
                      'entity_pos': self.entity_pos[index],
                      'head_enttype': self.head_enttype[index], 'tail_enttype': self.tail_enttype[index],
                      'enttype_length': self.enttype_length[index],
                      }
        if self.mode == self.MODE_INSTANCE:
            batch_data.update({'label': self.data_label[index]})
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            batch_data.update({'label': self.data_label[index][scope[:-1]]})
            batch_data.update({'instance_label': self.data_label[index]})
        if hasattr(self, 'scope') and scope is not None:
            batch_data.update({'scope': scope})
        if hasattr(self, 'weights_table'):
            batch_data.update({'weights': self.get_weights(batch_data['label'])})
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
        batch_data['entity_pos'] = np.concatenate([batch_data['entity_pos'], self.entity_pos[b:e]]) \
            if batch_data.__contains__('entity_pos') else self.entity_pos[b:e]
        batch_data['head_enttype'] = np.concatenate([batch_data['head_enttype'], self.head_enttype[b:e]]) \
            if batch_data.__contains__('head_enttype') else self.head_enttype[b:e]
        batch_data['tail_enttype'] = np.concatenate([batch_data['tail_enttype'], self.tail_enttype[b:e]]) \
            if batch_data.__contains__('tail_enttype') else self.tail_enttype[b:e]
        batch_data['enttype_length'] = np.concatenate([batch_data['enttype_length'], self.enttype_length[b:e]]) \
            if batch_data.__contains__('enttype_length') else self.enttype_length[b:e]

    def batch_padding(self, batch_data, padding):
        if padding > 0:
            batch_data['word'] = np.concatenate([batch_data['word'], np.zeros((padding, FLAGS.max_length), np.int32)])
            batch_data['pos1'] = np.concatenate([batch_data['pos1'], np.zeros((padding, FLAGS.max_length), np.int32)])
            batch_data['pos2'] = np.concatenate([batch_data['pos2'], np.zeros((padding, FLAGS.max_length), np.int32)])
            # if self.data_require['mask']:
            #     batch_data['mask'] = np.concatenate([batch_data['mask'], np.zeros((padding, FLAGS.max_length),
            #                                                                       np.int32)])
            if self.data_require['length']:
                batch_data['length'] = np.concatenate([batch_data['length'], np.zeros(padding, dtype=np.int32)])
            if self.data_require['label']:
                batch_data['label'] = np.concatenate([batch_data['label'], np.zeros(padding, dtype=np.int32)])
            if self.data_require['entity_pos']:
                batch_data['entity_pos'] = np.concatenate([batch_data['entity_pos'], np.zeros((padding, 2), np.int32)])
            if self.data_require['head_enttype']:
                batch_data['head_enttype'] = np.concatenate([batch_data['head_enttype'],
                                                             np.zeros((padding, FLAGS.et_max_length), np.int32)])
            if self.data_require['tail_enttype']:
                batch_data['tail_enttype'] = np.concatenate([batch_data['tail_enttype'],
                                                             np.zeros((padding, FLAGS.et_max_length), np.int32)])
            if self.data_require['enttype_length']:
                batch_data['enttype_length'] = np.concatenate([batch_data['enttype_length'],
                                                               np.zeros((padding, 2), np.int32)])

    def next_batch(self, batch_size):
        if self.begin >= len(self.order):
            self.begin = 0
            if self.shuffle:
                random.shuffle(self.order)
            raise StopIteration
        end = self.begin + batch_size
        if end > len(self.order):
            end = len(self.order)
        batch_data = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'length': [], 'label': [], 'entity_pos': [],
                      'head_enttype': [], 'tail_enttype': [], 'enttype_length': []}
        if self.mode == self.MODE_INSTANCE:
            batch_data['word'] = self.data_word[self.begin:end]
            batch_data['pos1'] = self.data_pos1[self.begin:end]
            batch_data['pos2'] = self.data_pos2[self.begin:end]
            # if self.data_require['mask']:
            #     batch_data['mask'] = self.data_mask[self.begin:end]
            if self.data_require['mask'] or self.data_require['length']:
                batch_data['length'] = self.data_length[self.begin:end]
            if self.data_require['label']:
                batch_data['label'] = self.data_label[self.begin:end]
            if self.data_require['mask'] or self.data_require['entity_pos']:
                batch_data['entity_pos'] = self.entity_pos[self.begin:end]
            if self.data_require['head_enttype']:
                batch_data['head_enttype'] = self.head_enttype[self.begin:end]
            if self.data_require['tail_enttype']:
                batch_data['tail_enttype'] = self.tail_enttype[self.begin:end]
            if self.data_require['enttype_mask'] or self.data_require['enttype_length']:
                batch_data['enttype_length'] = self.enttype_length[self.begin:end]
        elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
            if self.data_require['instance_label']:
                batch_data['instance_label'] = []
            if self.data_require['scope']:
                batch_data['scope'] = [0]
            batch_data['multi_label'] = []
            batch_data['entpair'] = []
            cur_pos = 0
            for k in range(self.begin, end):
                b, e = self.scope[self.order[k]][0], self.scope[self.order[k]][1]
                batch_data['word'].append(self.data_word[b:e])
                batch_data['pos1'].append(self.data_pos1[b:e])
                batch_data['pos2'].append(self.data_pos2[b:e])
                # if self.data_require['mask']:
                #     batch_data['mask'].append(self.data_mask[b:e])
                if self.data_require['mask'] or self.data_require['length']:
                    batch_data['length'].append(self.data_length[b:e])
                if self.data_require['label']:
                    batch_data['label'].append(self.data_label[b])
                if self.data_require['instance_label']:
                    batch_data['instance_label'].append(self.data_label[b:e])
                if self.data_require['mask'] or self.data_require['entity_pos']:
                    batch_data['entity_pos'].append(self.entity_pos[b:e])
                if self.data_require['scope']:
                    batch_data['scope'].append(batch_data['scope'][cur_pos] + e - b)
                if self.data_require['head_enttype']:
                    batch_data['head_enttype'].append(self.head_enttype[b:e])
                if self.data_require['tail_enttype']:
                    batch_data['tail_enttype'].append(self.tail_enttype[b:e])
                if self.data_require['enttype_mask'] or self.data_require['enttype_length']:
                    batch_data['enttype_length'].append(self.enttype_length[b:e])
                if self.mode == self.MODE_ENTPAIR_BAG:
                    _one_multi_label = np.zeros(self.rel_tot, dtype=np.int32)
                    _one_multi_label[self.data_label[b:e]] = 1
                    # noinspection PyTypeChecker
                    batch_data['multi_label'].append(_one_multi_label)
                    batch_data['entpair'].append(self.scope_name[self.order[k]])
                cur_pos += 1

            batch_data['word'] = np.concatenate(batch_data['word'])
            batch_data['pos1'] = np.concatenate(batch_data['pos1'])
            batch_data['pos2'] = np.concatenate(batch_data['pos2'])
            # if self.data_require['mask']:
            #     batch_data['mask'] = np.concatenate(batch_data['mask'])
            if self.data_require['mask'] or self.data_require['length']:
                batch_data['length'] = np.concatenate(batch_data['length'])
            if self.data_require['label']:
                batch_data['label'] = np.stack(batch_data['label'])
            if self.data_require['instance_label']:
                batch_data['instance_label'] = np.concatenate(batch_data['instance_label'])
            if self.data_require['mask'] or self.data_require['entity_pos']:
                batch_data['entity_pos'] = np.concatenate(batch_data['entity_pos'])
            if self.data_require['scope']:
                batch_data['scope'] = np.stack(batch_data['scope'])
            if self.data_require['head_enttype']:
                batch_data['head_enttype'] = np.concatenate(batch_data['head_enttype'])
            if self.data_require['tail_enttype']:
                batch_data['tail_enttype'] = np.concatenate(batch_data['tail_enttype'])
            if self.data_require['enttype_mask'] or self.data_require['enttype_length']:
                batch_data['enttype_length'] = np.concatenate(batch_data['enttype_length'])
            if self.mode == self.MODE_ENTPAIR_BAG:
                batch_data['multi_label'] = np.stack(batch_data['multi_label'])
                batch_data['entpair'] = np.stack(batch_data['entpair'])
            if cur_pos < batch_size:
                padding = batch_size - cur_pos
                if self.data_require['instance_label']:
                    batch_data['instance_label'] = np.concatenate([batch_data['instance_label'],
                                                                   np.zeros(padding, dtype=np.int32)])
                if self.data_require['scope']:
                    batch_data['scope'] = np.append(batch_data['scope'],
                                                    range(batch_data['scope'][cur_pos] + 1,
                                                          batch_data['scope'][cur_pos] + 1 + padding))
                if self.mode == self.MODE_ENTPAIR_BAG:
                    batch_data['multi_label'] = np.concatenate([batch_data['multi_label'],
                                                                [[0] * self.rel_tot] * padding])
                    batch_data['entpair'] = np.concatenate([batch_data['entpair'], ['None#None'] * padding])

        padding = batch_size - (end - self.begin)
        if self.data_require['mask']:
            bs = (len(batch_data['entity_pos']) + padding) if padding > 0 else len(batch_data['entity_pos'])
            batch_data['mask'] = np.zeros((bs, FLAGS.max_length), np.int32)
            for i in range(len(batch_data['entity_pos'])):
                h_pos = batch_data['entity_pos'][i][0]
                t_pos = batch_data['entity_pos'][i][1]
                length = batch_data['length'][i]
                batch_data['mask'][i][0:h_pos + 1] = [1] * (h_pos + 1)
                batch_data['mask'][i][h_pos + 1: t_pos + 1] = [2] * (t_pos - h_pos)
                batch_data['mask'][i][t_pos + 1: length] = [3] * (length - 1 - t_pos)

        if self.data_require['enttype_mask']:
            bs = (len(batch_data['enttype_length']) + padding) if padding > 0 else len(batch_data['enttype_length'])
            batch_data['enttype_mask'] = np.zeros((bs, 2 * FLAGS.et_max_length), np.int32)
            for i in range(len(batch_data['enttype_length'])):
                h_len = batch_data['enttype_length'][i][0]
                t_len = batch_data['enttype_length'][i][1]
                batch_data['enttype_mask'][i][0:h_len] = [1] * h_len
                batch_data['enttype_mask'][i][FLAGS.et_max_length:FLAGS.et_max_length + t_len] = [2] * t_len

        self.batch_padding(batch_data, padding)
        if hasattr(self, 'weights_table'):
            batch_data.update({'weights': self.get_weights(batch_data['label'])})
        self.begin = end

        return batch_data

    # def next_batch(self, batch_size):
    #     if self.begin >= len(self.order):
    #         self.begin = 0
    #         if self.shuffle:
    #             random.shuffle(self.order)
    #         raise StopIteration
    #     end = self.begin + batch_size
    #     if end > len(self.order):
    #         end = len(self.order)
    #
    #     batch_data = {}
    #     if self.mode == self.MODE_INSTANCE:
    #         self.add2batch_data(batch_data, self.begin, end)
    #         batch_data['label'] = self.data_label[self.begin:end]
    #     elif self.mode == self.MODE_ENTPAIR_BAG or self.mode == self.MODE_RELFACT_BAG:
    #         scope = np.zeros(batch_size + 1, dtype=np.int32)
    #         multi_label = [] if self.mode == self.MODE_ENTPAIR_BAG else None
    #         entpair = [] if self.mode == self.MODE_ENTPAIR_BAG and self.shuffle else None
    #         cur_pos = 0
    #         for k in range(self.begin, end):
    #             b, e = self.scope[self.order[k]][0], self.scope[self.order[k]][1]
    #             scope[cur_pos + 1] = scope[cur_pos] + e - b
    #             cur_pos += 1
    #             if self.mode == self.MODE_ENTPAIR_BAG:
    #                 _one_multi_rel = np.zeros(self.rel_tot, dtype=np.int32)
    #                 for n in range(b, e):
    #                     _one_multi_rel[self.data_label[n]] = 1
    #                 multi_label.append(_one_multi_rel)
    #                 batch_data['multi_label'] = np.concatenate([batch_data['multi_label'], multi_label]) \
    #                     if batch_data.__contains__('multi_label') else multi_label
    #                 if entpair is not None:
    #                     entpair.append(self.scope_name[self.order[k]])
    #
    #             self.add2batch_data(batch_data, b, e)
    #             batch_data['label'] = np.concatenate([batch_data['label'], [self.data_label[b]]]) \
    #                 if batch_data.__contains__('label') else [self.data_label[b]]
    #             batch_data['instance_label'] = np.concatenate([batch_data['instance_label'], self.data_label[b:e]]) \
    #                 if batch_data.__contains__('instance_label') else self.data_label[b:e]
    #         if self.mode == self.MODE_ENTPAIR_BAG:
    #             batch_data['entpair'] = self.scope_name[self.begin:end] if entpair is None else entpair
    #         if cur_pos < batch_size:
    #             padding = batch_size - cur_pos
    #             batch_data['instance_label'] = np.concatenate([batch_data['instance_label'],
    #                                                            np.zeros(padding, dtype=np.int32)])
    #             scope[cur_pos + 1:batch_size + 1] = range(scope[cur_pos] + 1, scope[cur_pos] + 1 + padding)
    #             if self.mode == self.MODE_ENTPAIR_BAG:
    #                 batch_data['entpair'] = np.concatenate([batch_data['entpair'], ['None#None'] * padding])
    #         batch_data['scope'] = scope
    #     self.batch_padding(batch_data, self.begin, end, batch_size)
    #
    #     if hasattr(self, 'weights_table'):
    #         batch_data.update({'weights': [self.weights_table[label] for label in batch_data['label']]})
    #     self.begin = end
    #
    #     return batch_data


class npy_data_loader(file_data_loader):
    def __init__(self, prefix=file_data_loader.TRAIN_PREFIX, mode=file_data_loader.MODE_RELFACT_BAG,
                 reprocess=False, shuffle=True):
        super().__init__('.npy', prefix, mode, reprocess, shuffle)

    @staticmethod
    def load_file(file_name):
        return np.load(file_name)

    def save_file(self, data, file_name):
        self.np_save_file(data, file_name)


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
        word_vec: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        rel2id: Json file storing relation-to-id diction in the following format
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

    def save_file(self, data, file_name):
        self.json_save_file(data, file_name)
