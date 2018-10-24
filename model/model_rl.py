import tensorflow as tf

import model.model_base as mb

FLAGS = tf.flags.FLAGS


def _set_params(*args):
    ovs = FLAGS.en, FLAGS.se, FLAGS.hidden_size
    FLAGS.en, FLAGS.se, FLAGS.hidden_size = args

    return ovs


class model_rl(mb.model):
    def __init__(self, data_loader, is_training=True):
        super().__init__(data_loader, is_training)

    def _network(self):
        if self.is_training:
            ovs = _set_params('cnn', 'instance', 2)
            super()._network()

            self.policy_agent_encoder = self.encoder
            self.policy_agent_logit = self.logit
            self.policy_agent_repre = self.repre
            self.policy_agent_output = self.output
            self.policy_agent_loss = self.loss
            self.policy_agent_global_step = tf.Variable(0, name='policy_agent_global_step', trainable=False)
            policy_agent_optimizer = mb.optimizer(FLAGS.learning_rate)
            policy_agent_grads_vars = policy_agent_optimizer.compute_gradients(self.policy_agent_loss)
            self.policy_agent_op = policy_agent_optimizer.apply_gradients(policy_agent_grads_vars,
                                                                          global_step=self.policy_agent_global_step)

            _set_params(*ovs)
        super()._network()
