import tensorflow as tf

from model.model_base import model

FLAGS = tf.flags.FLAGS


class model_rl(model):
    def __init__(self, data_loader, is_training=True):
        super().__init__(data_loader, is_training)

    def _network(self):
        super()._network()
