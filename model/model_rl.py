import tensorflow as tf

from model.model_base import model

FLAGS = tf.flags.FLAGS


class model_rl(model):
    def __init__(self, data_loader, is_training=True):
        super().__init__(data_loader, is_training)

    def _network(self):
        if self.is_training:
            ovs = self._set_params('cnn', 'instance', 2)
            super()._network()
            self.policy_agent_loss = self.loss
            self._set_params(*ovs)
        super()._network()

    def _set_params(self, *args):
        ovs = FLAGS.en, FLAGS.se, self.hidden_size
        FLAGS.en, FLAGS.se, self.hidden_size = args

        return ovs
