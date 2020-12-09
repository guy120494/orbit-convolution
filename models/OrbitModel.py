import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Dense, Flatten, MaxPooling1D, Conv1D
from tensorflow.python.keras.models import Model

from layers.OrbitConvolution import OrbitConvolution


class OrbitModel(Model):

    def __init__(self, axis: int = 0):
        super(OrbitModel, self).__init__()
        self.first_cnn_layer = OrbitConvolution(num_filters=32, kernel_size=3, axis=axis)
        self.second_cnn_layer = Conv1D(64, kernel_size=3)
        self.max_pooling = MaxPooling1D(pool_size=2)
        self.first_dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.first_dense = Dense(128, activation='relu')
        self.second_dropout = Dropout(0.5)
        self.second_dense = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.first_cnn_layer(inputs)
        x = self.second_cnn_layer(x)
        x = tf.nn.relu(x)
        x = self.max_pooling(x)
        # x = self.first_dropout(x, training=training)
        x = self.flatten(x)
        x = self.first_dense(x)
        # x = self.second_dropout(x, training=training)
        return self.second_dense(x)

    def get_config(self):
        pass
