import tensorflow as tf

import model.model_base as mb

FLAGS = tf.flags.FLAGS


class model_rl(mb.model):
    def __init__(self, data_loader, is_training=True):
        super().__init__(data_loader, is_training)

    def _network(self):
        if self.is_training:
            ovs = self._set_params('cnn', 'instance', 2)
            super()._network()

            self.policy_agent_encoder = self.encoder
            self.policy_agent_logit = self.logit
            self.policy_agent_repre = self.repre
            self.policy_agent_output = self.output
            self.policy_agent_loss = self.loss
            self.policy_agent_global_step = tf.Variable(0, name='policy_agent_global_step', trainable=False)
            self.policy_agent_op = None

            self._set_params(*ovs)
        super()._network()

    def _set_params(self, *args):
        ovs = FLAGS.en, FLAGS.se, self.rel_tot
        FLAGS.en, FLAGS.se, self.rel_tot = args

        return ovs
