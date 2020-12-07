from tensorflow.python.keras.models import Model
import tensorflow as tf
from layers.OrbitConvolution import OrbitConvolution


class BasicModel(Model):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.first_layer = OrbitConvolution(kernel_size=3, num_filters=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.first_layer(inputs)
        x = self.flatten(x)
        return self.dense(x)

    def get_config(self):
        pass
