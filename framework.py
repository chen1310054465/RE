import math
import os
import sys
import time

import numpy as np
import sklearn.metrics
import tensorflow as tf

FLAGS = tf.flags.FLAGS


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
    MODE_BAG = 0  # Train and test the model at bag level.
    MODE_INS = 1  # Train and test the model at instance level

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

    def _summary(self, labels, outputs):
        for i, label in enumerate(labels):
            self.acc_total.add(outputs[i] == label)
            if label != 0:
                self.acc_not_na.add(outputs[i] == label)

    def _one_step_multi_models(self, models, run_array, return_label=True):
        batch_label = []
        result = None
        for model in models:
            batch_data = self.train_data_loader.next_batch(self.train_data_loader.batch_size // len(models))
            result = self._one_step(model, batch_data, run_array, model.get_weights(batch_data['label']))
            batch_label.append(batch_data['label'])
            merged_summary = self.sess.run(tf.summary.merge_all(), feed_dict=self.feed_dict)
            self.summary_writer.add_summary(merged_summary, self.step)

        batch_label = np.concatenate(batch_label)
        if return_label:
            result += [batch_label]
        return result

    def _one_step(self, model, batch_data, run_array, weights=None):
        self.feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.length: batch_data['length'],
            model.label: batch_data['label'],
            model.instance_label: batch_data['instance_label'],
            model.scope: batch_data['scope'],
        }
        if 'mask' in batch_data and model.mask is not None:  # hasattr(model, "mask"):
            self.feed_dict.update({model.mask: batch_data['mask']})
        if weights is not None:
            self.feed_dict.update({model.weights: weights})
        return self.sess.run(run_array, self.feed_dict)

    def train(self, model, optimizer=tf.train.GradientDescentOptimizer):
        assert (self.train_data_loader.batch_size % FLAGS.gn == 0)
        print("Start training...")

        # Init
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        optimizer = optimizer(FLAGS.learning_rate)

        # Multi GPUs
        tower_grads = []
        tower_models = []
        for gpu_id in range(FLAGS.gn):
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):
                    cur_model = model(self.train_data_loader)
                    tower_grads.append(optimizer.compute_gradients(cur_model.loss))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss", cur_model.loss)
                    tf.add_to_collection("train_output", cur_model.output)

        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)
        output_collection = tf.get_collection("train_output")
        output = tf.concat(output_collection, 0)

        grads = average_gradients(tower_grads)
        train_op = optimizer.apply_gradients(grads)
        # summary writer
        self.summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, self.sess.graph)
        # saver
        self.saver = tf.train.Saver(max_to_keep=None)
        if FLAGS.pm is not None:
            self.saver.restore(self.sess, FLAGS.pm)
        else:
            self.sess.run(tf.global_variables_initializer())
        # Training
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0  # Stop training after several epochs without improvement.
        for epoch in range(FLAGS.max_epoch):
            print('###### epoch ' + str(epoch) + ' ######')
            self.acc_not_na.clear()
            self.acc_total.clear()
            self.step = 0
            time_sum = 0
            while True:
                time_start = time.time()
                try:
                    iter_loss, iter_output, _train_op, iter_label = self._one_step_multi_models(tower_models,
                                                                                                [loss, output,
                                                                                                 train_op])
                except StopIteration:
                    break
                time_end = time.time()
                t = time_end - time_start
                time_sum += t

                for i, label in enumerate(iter_label):
                    self.acc_total.add(iter_output[i] == label)
                    if label != 0:
                        self.acc_not_na.add(iter_output[i] == label)

                if self.acc_not_na.total > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (
                        epoch, self.step, t, iter_loss, self.acc_not_na.get(), self.acc_total.get()))
                    sys.stdout.flush()
                self.step += 1
            print("\nAverage iteration time: %f" % (time_sum / self.step))

            if (epoch + 1) % FLAGS.test_epoch == 0:
                metric = self.test(model)
                if metric > best_metric:
                    best_metric = metric
                    best_prec = self.cur_prec
                    best_recall = self.cur_recall
                    print("Best model, storing...")
                    if not os.path.isdir(FLAGS.ckpt_dir):
                        os.mkdir(FLAGS.ckpt_dir)
                    path = self.saver.save(self.sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
                    print("Finish storing, saved path: " + path)
                    not_best_count = 0
                else:
                    not_best_count += 1

            if not_best_count >= 20:
                break

        print("######")
        print("Finish training " + FLAGS.model_name)
        print("Best epoch auc = %f" % best_metric)
        if (not best_prec is None) and (not best_recall is None):
            if not os.path.isdir(FLAGS.test_result_dir):
                os.mkdir(FLAGS.test_result_dir)
            np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_x.npy"), best_recall)
            np.save(os.path.join(FLAGS.test_result_dir, FLAGS.model_name + "_y.npy"), best_prec)

    def test(self, model, model_name=None, return_result=False, mode=MODE_BAG):
        if mode == framework.MODE_BAG:
            return self._test_bag(model, model_name, return_result=return_result)
        elif mode == framework.MODE_INS:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _test_bag(self, model, model_name=None, return_result=False):
        print("Testing...")
        if self.sess is None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, is_training=False)
        if not model_name is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(FLAGS.ckpt_dir, model_name))

        self.acc_total.clear()
        self.acc_not_na.clear()
        entpair_tot = 0
        test_result = []
        pred_result = []

        for i, batch_data in enumerate(self.test_data_loader):
            iter_logit, iter_output = self._one_step(model, batch_data, [model.logit, model.output])
            self._summary(batch_data['label'], iter_output)

            if self.acc_not_na.total > 0:
                sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (
                    i, self.acc_not_na.get(), self.acc_total.get()))
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
    def pretrain_policy_agent(self, model, max_epoch=1):
        for epoch in range(max_epoch):
            print(('[pretrain policy agent] ' + 'epoch ' + str(epoch) + ' starts...'))
            self.acc_total.clear()
            self.acc_not_na.clear()

            for i, batch_data in enumerate(self.train_data_loader):
                policy_agent_label = batch_data['label'] + 0
                policy_agent_label[policy_agent_label > 0] = 1
                weights = np.ones(policy_agent_label.shape, dtype=np.float32)
                iter_output, iter_loss = self._one_step(model, batch_data,
                                                        [model.policy_agent_output, model.policy_agent_loss,
                                                         model.policy_agent_op, model.policy_agent_global_step],
                                                        weights=weights)[:2]

                self._summary(batch_data['label'], iter_output)

                sys.stdout.write(
                    "[pretrain policy agent] epoch %d step %d | loss : %f, accuracy: %f, accuracy of 1: %f" % (
                        epoch, i, iter_loss, self.acc_total.get(), self.acc_not_na.get()) + '\n')
                sys.stdout.flush()

            if self.acc_total.get() > 0.9:
                break

    def train_rl(self, model, max_epoch=1, mode=MODE_BAG):
        for epoch in range(max_epoch):
            print(('epoch ' + str(epoch) + ' starts...'))
            self.acc_not_na.clear()
            self.acc_total.clear()
            self.step = 0

            # update policy agent
            tot_delete = 0
            batch_count = 0
            action_result_his = []
            reward = 0.0
            for i, batch_data in enumerate(self.train_data_loader):
                # make action
                action_result, batch_loss = self._one_step(model, batch_data, [model.policy_agent_output, model.loss])
                action_result_his += action_result

                # calculate reward
                batch_label = batch_data['label']
                batch_delete = np.sum(np.logical_and(batch_label != 0, action_result == 0))
                batch_label[action_result == 0] = 0

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
                        batch_result = np.take(action_result_his, index)
                        weights = np.ones(batch_result.shape, dtype=np.float32)
                        weights *= reward
                        weights *= alpha
                        iter_output, iter_loss = self._one_step(model, batch_data,
                                                                [model.policy_agent_output, model.policy_agent_loss,
                                                                 model.policy_agent_op, model.policy_agent_global_step],
                                                                weights=weights)[:2]

                        for k, label in enumerate(batch_data['label']):
                            self.acc_total.add(iter_output[k] == label)
                            if label != 0:
                                self.acc_not_na.add(iter_output[k] == label)

                        sys.stdout.write(
                            "[pretrain policy agent] epoch %d step %d | loss : %f, accuracy: %f, accuracy of 1: %f" % (
                                epoch, i, iter_loss, self.acc_total.get(), self.acc_not_na.get()) + '\n')
                        sys.stdout.flush()
                    if self.acc_total.get() > 0.9:
                        break

                    batch_count = 0
                    reward = 0
                    tot_delete = 0

            for i, batch_data in enumerate(self.train_data_loader):
                weights = model.get_weights(batch_data['label'])
                if mode == framework.MODE_BAG:
                    # make action
                    action_result, outputs, loss = self._one_step(model, batch_data,
                                                                  [model.policy_agent_output, model.output, model.loss],
                                                                  weights=weights)
                    # calculate reward
                    batch_label = batch_data['label']
                    # batch_delete = np.sum(np.logical_and(batch_label != 0, action_result == 0))
                    batch_label[action_result == 0] = 0
                else:
                    index = list(range(i * FLAGS.batch_size, (i + 1) * FLAGS.batch_size))
                    for j in index:
                        weights.append(model.get_weights(batch_data['label'])[j])
                    outputs, loss = self._one_step(model, batch_data, [model.output, model.loss], weights=weights)[0]

                self._summary(batch_data['label'], outputs)
                sys.stdout.write(
                    "epoch %d step %d | loss : %f, not NA accuracy: %f, total accuracy %f" % (
                        epoch, i, loss, self.acc_not_na.get(), self.acc_total.get()) + '\r')
                sys.stdout.flush()

            if (epoch + 1) % FLAGS.save_epoch == 0:
                print(('epoch ' + str(epoch + 1) + ' has finished'))
                print('saving model...')
                path = self.saver.save(self.sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name),
                                       global_step=epoch)
                print(('have saved model to ' + path))

    # rl part ends
