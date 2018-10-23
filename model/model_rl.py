import tensorflow as tf

from model.model_base import model

FLAGS = tf.flags.FLAGS


def _set_params(*args):
    ovs = FLAGS.en, FLAGS.se, FLAGS.hidden_size
    FLAGS.en, FLAGS.se, FLAGS.hidden_size = args

    return ovs


class model_rl(model):
    def __init__(self, data_loader, is_training=True):
        super().__init__(data_loader, is_training)

    def _network(self):
        if self.is_training:
            ovs = _set_params('cnn', 'instance', 2)
            super()._network()

            self.policy_agent_logit = self.logit
            self.policy_agent_output = self.output
            self.policy_agent_loss = self.loss
            self.policy_agent_global_step = tf.Variable(0, name='policy_agent_global_step', trainable=False)
            self.policy_agent_op = None

            _set_params(*ovs)
        super()._network()
