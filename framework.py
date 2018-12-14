import math
import os
import re
import sys
import time
import gc

import numpy as np
import sklearn.metrics
import tensorflow as tf
from data_loader import file_data_loader

FLAGS = tf.flags.FLAGS
# define some configurable parameter
tf.flags.DEFINE_string('dn', 'nyt', 'dataset name')
tf.flags.DEFINE_string('en', 'pcnn', 'encoder')
tf.flags.DEFINE_string('se', 'att', 'selector')
tf.flags.DEFINE_string('cl', 'softmax', 'classifier')
tf.flags.DEFINE_string('ac', 'relu', 'activation')
tf.flags.DEFINE_string('op', 'sgd', 'optimizer')
tf.flags.DEFINE_integer('et', 0, 'whether to add entity type info')
tf.flags.DEFINE_integer('ad', 0, 'whether to add adversarial training')
tf.flags.DEFINE_integer('gn', 1, 'gpu_nums')
tf.flags.DEFINE_integer('pm', 0, 'whether to pretrain model')
# define some specified parameter
tf.flags.DEFINE_integer('max_epoch', 120, 'max epoch')
tf.flags.DEFINE_integer('save_epoch', 1, 'save epoch')
tf.flags.DEFINE_integer('hidden_size', 230, 'hidden size')
tf.flags.DEFINE_integer('f1', 230, 'filter1 size')
tf.flags.DEFINE_integer('f2', 230, 'filter2 size')
tf.flags.DEFINE_integer('rnn_hidden_size', 230, 'rnn hidden size for rcnn model')
tf.flags.DEFINE_integer('cnn_hidden_size', 230, 'cnn hidden size for rcnn model')
tf.flags.DEFINE_integer('et_hidden_size', 80, 'entity type hidden size')
tf.flags.DEFINE_integer('ib_num', 3, 'num of identity block for residual network')
tf.flags.DEFINE_integer('batch_size', 160, 'batch size')
tf.flags.DEFINE_integer('max_length', 120, 'word max length')
tf.flags.DEFINE_integer('et_max_length', 100, 'entity type max length')
tf.flags.DEFINE_integer('word_dim', 50, 'word embedding dimensionality')
tf.flags.DEFINE_integer('pos_dim', 5, 'pos embedding dimensionality')
tf.flags.DEFINE_integer('et_dim', 12, 'entity type embedding dimensionality')
tf.flags.DEFINE_integer('et_concat_axis', 1, 'which axis head and tail entity type embedding concat at')
tf.flags.DEFINE_integer('et_half', 0, ' whether half of dataset use entity type embedding')
tf.flags.DEFINE_integer('li_encoder_mode', 0, 'organize encoder mode')
tf.flags.DEFINE_float('learning_rate', 0.5, 'learning rate')
tf.flags.DEFINE_string('et_en', 'pcnn', 'entity type encoder')
tf.flags.DEFINE_string('ckpt_dir', os.path.join('checkpoint', FLAGS.dn), 'checkpoint dir')
tf.flags.DEFINE_string('summary_dir', os.path.join('summary', FLAGS.dn), 'summary dir')
tf.flags.DEFINE_string('test_result_dir', os.path.join('test_result', FLAGS.dn), 'test result dir')
tf.flags.DEFINE_string('dataset_dir', os.path.join('origin_data', FLAGS.dn), 'origin dataset dir')
tf.flags.DEFINE_string('processed_data_dir', os.path.join('processed_data', FLAGS.dn), 'processed data dir')
tf.flags.DEFINE_string('model_name', (FLAGS.dn + '_' + ('et' + (FLAGS.et_en[0] + '_' if FLAGS.et_en in ['pcnn', 'dense']
                                                        else 'c_') if FLAGS.et else '') +  # dataset_name entity_type
                                        ('half_' if FLAGS.et_half else '') +
                                      ('li_' if FLAGS.li_encoder_mode and (FLAGS.et or re.search("r.*cnn", FLAGS.en))
                                       else '') + FLAGS.en + "_" + FLAGS.se +  # encoder selector
                                      (('_' + FLAGS.cl) if FLAGS.cl != 'softmax' else '') +  # classifier
                                      (('_' + FLAGS.ac) if FLAGS.ac != 'relu' else '') +  # activation
                                      (('_' + FLAGS.op) if FLAGS.op != 'sgd' else '') +  # optimizer
                                      ('_ad' if FLAGS.ad else '')), 'model_name')  # adversarial_training

activations = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh,
               'relu': tf.nn.relu, 'leaky_relu': tf.nn.leaky_relu}
optimizers = {'sgd': tf.train.GradientDescentOptimizer, 'momentum': tf.train.MomentumOptimizer,
              'adagrad': tf.train.AdagradOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
              'adam': tf.train.AdamOptimizer}


def init(is_training=True):
    if 'help' in sys.argv:
        print('Usage: python3 ' + sys.argv[0] + ' [--dn dataset_name] [--et: ent_type] [--en encoder] [--se selector]'
              + ('[--cl classifier] [--ac activation] [--op optimizer] [--ad adversarial_training]\n       '
                 + '[--gn gpu_nums] [--pm pretrain_model] [--max_epoch] [--save_epoch] '
                 + '[--learning_rate] ' if is_training else '')
                 + '[--hidden_size] [--et_hidden_size] [--rnn_hidden_size] [--cnn_hidden_size]\n       '
                 + '[--word_dim] [--pos_dim] [--et_en entity type encoder] [--et_dim] [--et_concat_axis] [--et_half] '
                 + '[--f1] [--f2] [--ib_num] [--batch_size] [--li_encoder_mode]')
        print('**************************************args details**********************************************')
        print('**  --dn: (dataset_name)[nyt(New York Times dataset)...], put it in origin_data dir           **')
        print('**  --et: (ent_type)whether to add entity type info(default 0), 0(no), 1(yes)                 **')
        print('**  --en: (encoder)[cnn pcnn rnn birnn rnn_lstm birnn_lstm rnn_gru birnn_gru]                 **')
        print('**                 [rcnn rpcnn bircnn birpcnn rcnn_lstm rpcnn_lstm bircnn_lstm birpcnn_lstm]  **')
        print('**                 [rcnn_gru rpcnn_gru bircnn_gru birpcnn_gru]                                **')
        print('**  --se: (selector)[instance att one ave cross_max att_rl one_rl ave_rl cross_max_rl]        **')
        if is_training:
            print('**  --cl: (classifier)[softmax sigmoid soft_label]                                            **')
            print('**  --ac: (activation)' + str([act for act in activations]) + '                               **')
            print('**  --op: (optimizer)' + str([op for op in optimizers]) + '                       **')
            print('**  --ad: (adversarial_training)whether to perturb while training(default 0), 0(no), 1(yes)   **')
            print('**  --gn: (gpu_nums)denotes num of gpu for training(default 1)                                **')
            print('**  --pm: (pretrain_model)whether to pretrain model(default 0), 0(no), 1(yes)                 **')
            print('**  --max_epoch: max epoch util stopping training(default 120)                                **')
            print('**  --save_epoch: how many epoch to save result while training(default 2)                     **')
            print('**  --learning_rate: learning rate(default 0.5 when training, whereas 1 when testing)         **')
        print('**  --hidden_size: hidden size of encoder(default 230)                                        **')
        print('**  --et_hidden_size: hidden size of entity type encoder(default 80)                          **')
        print('**  --rnn_hidden_size: hidden size of rnn encoder for rcnn model(default 230)                 **')
        print('**  --cnn_hidden_size: hidden size of cnn encoder for rcnn model(default 230)                 **')
        print('**  --word_dim: word embedding dimensionality(default 50)                                     **')
        print('**  --pos_dim: pos embedding dimensionality(default 5)                                        **')
        print('**  --et_en: (entity type encoder)[cnn pcnn dense](default cnn)                               **')
        print('**  --et_dim: entity type embedding dimensionality(default 12)                                **')
        print('**  --et_concat_axis: [-1, 1], which axis head and tail et_embedding concat(default 1)        **')
        print('**  --et_half: whether half of dataset use entity type embedding(default 1)                   **')
        print('**  --f1: filter1 size for residual network(default 128)                                      **')
        print('**  --f2: filter1 size for residual network(default 230)                                      **')
        print('**  --ib_num: num of identity block for residual network(default 4)                           **')
        print('**  --batch_size: batch size of corpus for each step(default 160)                             **')
        print('**  --li_encoder_mode: organize encoder mode(0 denotes concat, 1 denotes linear transform)    **')
        print('************************************************************************************************')
        exit()
    if FLAGS.et:
        assert FLAGS.et_concat_axis in [-1, 1]


def dropout(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if len(grads) == 0:
            continue
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        self.total += 1
        if self.total == 0:
            return 0
        return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class framework:
    def __init__(self, train_data_loader=None, test_data_loader=None):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None
        self.saver = None
        self.summary_writer = None
        self.feed_dict = {}
        self.step = 0
        self.acc_not_na = accuracy()
        self.acc_total = accuracy()
        self.activation = tf.nn.relu
        self.optimizer = tf.train.GradientDescentOptimizer
        if FLAGS.ac in activations:
            self.activation = activations[FLAGS.ac]

        if FLAGS.op in optimizers:
            self.optimizer = optimizers[FLAGS.op]

    def _summary(self, labels, outputs):
        for i, label in enumerate(labels):
            self.acc_total.add(outputs[i] == label)
            if label != 0:
                self.acc_not_na.add(outputs[i] == label)

    def _one_step_multi_models(self, models, run_array, return_label=True):
        batch_label = []
        result = None
        for model in models:
            batch_data = self.train_data_loader.next_batch(FLAGS.batch_size // len(models))
            result = self._one_step(model, batch_data, run_array)
            batch_label.append(batch_data['label'])

        batch_label = np.concatenate(batch_label)
        if return_label:
            result += [batch_label]
        return result

    def _one_step(self, model, batch_data, run_array, weights=None, fd_updater=None):
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
        }
        if model.mask is not None:  # hasattr(model, "mask"):
            feed_dict.update({model.mask: batch_data['mask']})
        if model.length is not None:
            feed_dict.update({model.length: batch_data['length']})
        if model.label is not None:
            feed_dict.update({model.label: batch_data['label']})
        if model.instance_label is not None:
            feed_dict.update({model.instance_label: batch_data['instance_label']})
        if model.entity_pos is not None:
            feed_dict.update({model.entity_pos: batch_data['entity_pos']})
        if model.scope is not None:
            feed_dict.update({model.scope: batch_data['scope']})
        if model.is_training:
            weights = batch_data['weights'] if weights is None else weights
            feed_dict.update({model.weights: weights})
        if model.head_enttype is not None:
            feed_dict.update({model.head_enttype: batch_data['head_enttype']})
        if model.tail_enttype is not None:
            feed_dict.update({model.tail_enttype: batch_data['tail_enttype']})
        if model.enttype_length is not None:
            feed_dict.update({model.enttype_length: batch_data['enttype_length']})
        if model.enttype_mask is not None:
            feed_dict.update({model.enttype_mask: batch_data['enttype_mask']})

        if fd_updater is not None:
            fd_updater(feed_dict)
        if self.step % 50 == 0:
            # merged_summary = self.sess.run(tf.summary.merge_all(), feed_dict=feed_dict)
            # self.summary_writer.add_summary(merged_summary, self.step)
            gc.collect()

        return self.sess.run(run_array, feed_dict)

    def train(self, model):
        assert (FLAGS.batch_size % FLAGS.gn == 0)
        print("Start training...")

        # Init
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        optimizer = self.optimizer(FLAGS.learning_rate)

        # Multi GPUs
        tower_grads = []
        tower_models = []
        half_tower_grads = []
        half_tower_models = []
        for gpu_id in range(FLAGS.gn):
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):
                    cur_model = model(self.train_data_loader, et=FLAGS.et, activation=self.activation)
                    tower_grads.append(optimizer.compute_gradients(cur_model.loss))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss", cur_model.loss)
                    tf.add_to_collection("train_output", cur_model.output)
                    if FLAGS.et_half:
                        half_model = model(self.train_data_loader, activation=self.activation)
                        half_tower_grads.append(optimizer.compute_gradients(half_model.loss))
                        half_tower_models.append(half_model)
                        tf.add_to_collection("half_loss", half_model.loss)
                        tf.add_to_collection("half_train_output", half_model.output)

        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)
        output_collection = tf.get_collection("train_output")
        output = tf.concat(output_collection, 0)

        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)

        half_loss_collection = tf.get_collection("half_loss") if FLAGS.et_half else None
        half_loss = tf.add_n(half_loss_collection) / len(half_loss_collection) if FLAGS.et_half else None
        half_output_collection = tf.get_collection("half_train_output") if FLAGS.et_half else None
        half_output = tf.concat(half_output_collection, 0) if FLAGS.et_half else None

        half_grads = average_gradients(half_tower_grads) if FLAGS.et_half else None
        half_train_op = optimizer.apply_gradients(half_grads) if FLAGS.et_half else None

        if not os.path.exists(FLAGS.ckpt_dir):
            os.makedirs(FLAGS.ckpt_dir)
        summary_dir = os.path.join(FLAGS.summary_dir, FLAGS.model_name)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # summary writer
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        # saver
        self.saver = tf.train.Saver(max_to_keep=None)
        # Training
        best_metric = 0
        if FLAGS.pm:
            self.saver.restore(self.sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
            x_filename = os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_x.npy")
            y_filename = os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_y.npy")
            if os.path.exists(x_filename) and os.path.exists(y_filename):
                x = np.load(x_filename)
                y = np.load(y_filename)
                best_metric = sklearn.metrics.auc(x=x, y=y)
        else:
            self.sess.run(tf.global_variables_initializer())
        best_prec = None
        best_recall = None
        not_best_count = 0  # Stop training after several epochs without improvement.
        for epoch in range(FLAGS.max_epoch):
            print('###### epoch ' + str(epoch) + ' ######')
            self.acc_not_na.clear()
            self.acc_total.clear()
            self.step = 0
            time_sum = 0
            models, op, lo, out = half_tower_models, half_train_op, half_loss, half_output
            while True:
                time_start = time.time()
                try:
                    if FLAGS.et_half and self.step == self.train_data_loader.batch // 2:
                        models, op, lo, out = tower_models, train_op, loss, output
                        self.train_data_loader.data_require = models[0].data_require
                    iter_loss, iter_output, iter_label = self._one_step_multi_models(models, [op, lo, out])[1:]
                except StopIteration:
                    break
                time_end = time.time()
                t = time_end - time_start
                time_sum += t

                self._summary(iter_label, iter_output)
                if self.acc_not_na.total > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (
                        epoch, self.step, t, iter_loss, self.acc_not_na.get(), self.acc_total.get()))
                    sys.stdout.flush()
                self.step += 1
            gc.collect()
            print("\nAverage iteration time: %f" % (time_sum / self.step))

            for m in tower_models:
                if '_rl' in FLAGS.se:
                    self.pretrain_policy_agent(m, max_epoch=1)
                    self.train_rl(m, max_epoch=2)
                    self.train_data_loader.mode = file_data_loader.MODE_RELFACT_BAG

            if (epoch + 1) % FLAGS.save_epoch == 0:
                metric = self.test(model)
                if metric > best_metric:
                    best_metric = metric
                    best_prec = self.cur_prec
                    best_recall = self.cur_recall

                    print("Best model, storing...")
                    path = self.saver.save(self.sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
                    print("Finish storing, saved path: " + path)
                    not_best_count = 0
                else:
                    not_best_count += 1
                gc.collect()

            if not_best_count >= 20:
                break

        print("######")
        print("Finish training " + FLAGS.model_name)
        print("Best epoch auc = %f" % best_metric)
        if (not best_prec is None) and (not best_recall is None):
            if not os.path.exists(FLAGS.test_result_dir):
                os.makedirs(FLAGS.test_result_dir)
            np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_x.npy"), best_recall)
            np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_y.npy"), best_prec)

    def test(self, model, model_name=None, return_result=False, mode=file_data_loader.MODE_ENTPAIR_BAG):
        if mode == file_data_loader.MODE_ENTPAIR_BAG:
            return self._test_bag(model, model_name, return_result=return_result)
        elif mode == file_data_loader.MODE_INSTANCE:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _test_bag(self, model, model_name=None, return_result=False):
        print("Testing...")
        if self.sess is None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, et=FLAGS.et, activation=self.activation, is_training=False)
        if not model_name is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(FLAGS.ckpt_dir, model_name))

        self.acc_total.clear()
        self.acc_not_na.clear()
        entpair_tot = 0
        test_result = []
        pred_result = []

        for i, batch_data in enumerate(self.test_data_loader):
            time_start = time.time()
            iter_logit, iter_output = self._one_step(model, batch_data, [model.logit, model.output])
            time_end = time.time()
            t = time_end - time_start
            self._summary(batch_data['label'], iter_output)

            if self.acc_not_na.total > 0:
                sys.stdout.write("[TEST] step %d time %.2f | not NA accuracy: %f, accuracy: %f\r" % (
                    i, t, self.acc_not_na.get(), self.acc_total.get()))
                sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    test_result.append({'score': iter_logit[idx][rel], 'flag': batch_data['multi_label'][idx][rel]})
                    if batch_data['entpair'][idx] != "None#None":
                        pred_result.append({'score': float(iter_logit[idx][rel]),
                                            'entpair': batch_data['entpair'][idx], 'relation': rel})
                entpair_tot += 1

        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = []
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / self.test_data_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
        self.cur_prec = prec
        self.cur_recall = recall

        if not return_result:
            return auc
        else:
            return auc, pred_result

    # rl part
    def _policy_agent_one_step(self, model, batch_data, weights, fd_updater=None, eval_acc=True):
        iter_output, iter_loss = self._one_step(model, batch_data,
                                                [model.policy_agent_output, model.policy_agent_loss,
                                                 model.policy_agent_op, model.policy_agent_global_step],
                                                weights=weights, fd_updater=fd_updater)[:2]
        if eval_acc:
            self._summary(batch_data['label'], iter_output)
        return iter_output, iter_loss

    def _make_action(self, model, batch_data):
        fd_updater = lambda fd: {fd.__delitem__(model.label), fd.__delitem__(model.instance_label),
                                 fd.__delitem__(model.scope)}
        return self._one_step(model, batch_data, [model.policy_agent_output], fd_updater=fd_updater)

    def pretrain_policy_agent(self, model, mode=file_data_loader.MODE_INSTANCE, max_epoch=1):
        policy_agent_optimizer = self.optimizer(FLAGS.learning_rate)
        policy_agent_grads_vars = policy_agent_optimizer.compute_gradients(model.policy_agent_loss)
        model.policy_agent_op = policy_agent_optimizer.apply_gradients(policy_agent_grads_vars,
                                                                       global_step=model.policy_agent_global_step)
        self.train_data_loader.mode = mode
        for epoch in range(max_epoch):
            print(('[pretrain policy agent] ' + 'epoch ' + str(epoch) + ' starts...'))
            self.acc_total.clear()
            self.acc_not_na.clear()

            for i, batch_data in enumerate(self.train_data_loader):
                policy_agent_label = batch_data['label'] + 0
                policy_agent_label[policy_agent_label > 0] = 1
                weights = np.ones(policy_agent_label.shape, dtype=np.float32)
                iter_output, iter_loss = self._policy_agent_one_step(model, batch_data, weights)

                sys.stdout.write(
                    "[pretrain policy agent] epoch %d step %d | loss : %f, not NA accuracy: %f, accuracy: %f\r" % (
                        epoch, i, iter_loss, self.acc_not_na.get(), self.acc_total.get()))
                sys.stdout.flush()

            if self.acc_total.get() > 0.9:
                break

    def train_rl(self, model, max_epoch=1, mode=file_data_loader.MODE_RELFACT_BAG):
        self.train_data_loader.mode = mode
        for epoch in range(max_epoch):
            print(('epoch ' + str(epoch) + ' starts...'))
            self.acc_not_na.clear()
            self.acc_total.clear()
            self.step = 0

            # update policy agent
            tot_delete = 0
            batch_count = 0
            reward = 0.0
            action_result_his = []
            for i, batch_data in enumerate(self.train_data_loader):
                # make action
                action_result = self._make_action(model, batch_data)
                action_result_his = np.append(action_result_his, action_result)

                # calculate reward
                batch_label = batch_data['instance_label']
                batch_delete = np.sum(np.logical_and(batch_label != 0, action_result == 0))
                batch_label[action_result == 0] = 0

                batch_loss = self._one_step(model, batch_data, [model.loss])[0]
                reward += batch_loss
                tot_delete += batch_delete
                batch_count += 1

                alpha = 0.1
                if batch_count == 100:
                    reward = reward / float(batch_count)
                    average_loss = reward
                    reward = - math.log(1 - math.e ** (-reward))
                    sys.stdout.write(
                        'tot delete : %f | reward : %f | average loss : %f' % (tot_delete, reward, average_loss))
                    sys.stdout.flush()
                    for j in range(i - batch_count + 1, i + 1):
                        index = list(range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size))
                        batch_result = action_result_his[index]
                        weights = np.ones(batch_result.shape, dtype=np.float32)
                        weights *= reward
                        weights *= alpha

                        batch_data = self.train_data_loader.batch_data(index)
                        fd_updater = lambda fd: fd.update({model.label: batch_result})
                        iter_output, iter_loss = self._policy_agent_one_step(model, batch_data, weights, fd_updater)

                        sys.stdout.write(
                            "[pretrain policy agent] epoch %d step %d | loss : %f, not NA accuracy: %f, accuracy: "
                            "%f\r" % (epoch, i, iter_loss, self.acc_not_na.get(), self.acc_total.get()))
                        sys.stdout.flush()
                    if self.acc_total.get() > 0.9:
                        break

                    batch_count = 0
                    reward = 0
                    tot_delete = 0

            for i, batch_data in enumerate(self.train_data_loader):
                if mode == file_data_loader.MODE_RELFACT_BAG:
                    # make action
                    action_result = self._make_action(model, batch_data)
                    # calculate reward
                    batch_label = batch_data['instance_label']
                    # batch_delete = np.sum(np.logical_and(batch_label != 0, action_result == 0))
                    batch_label[action_result == 0] = 0
                loss, outputs = self._one_step(model, batch_data, [model.loss, model.output],
                                               model.get_weights(batch_data['label']))

                self._summary(batch_data['label'], outputs)
                sys.stdout.write(
                    "epoch %d step %d | loss : %f, not NA accuracy: %f, total accuracy %f\r" % (
                        epoch, i, loss, self.acc_not_na.get(), self.acc_total.get()))
                sys.stdout.flush()

            if (epoch + 1) % FLAGS.save_epoch == 0:
                print(('epoch ' + str(epoch + 1) + ' has finished'))
                print('saving model...')
                path = self.saver.save(self.sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name),
                                       global_step=epoch)
                print(('have saved model to ' + path))

    # rl part ends
