import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling1D
from tensorflow.python.keras.models import Model

from layers.OrbitConvolution import OrbitConvolution


class OrbitModel(Model):

    def __init__(self):
        super(OrbitModel, self).__init__()
        self.first_cnn_layer = None
        self.second_cnn_layer = OrbitConvolution(kernel_size=3, num_filters=64, axis=1)
        self.max_pooling = MaxPooling1D(pool_size=2)
        self.first_dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.first_dense = Dense(128, activation='relu')
        self.second_dropout = Dropout(0.5)
        self.second_dense = Dense(10, activation='softmax')

    def build(self, input_shape):
        self.first_cnn_layer = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.first_cnn_layer(inputs)
        x = self.second_cnn_layer(x)
        x = tf.nn.relu(x)
        x = self.max_pooling(x)
        x = self.first_dropout(x, training=training)
        x = self.flatten(x)
        x = self.first_dense(x)
        x = self.second_dropout(x, training=training)
        return self.second_dense(x)

    def get_config(self):
        pass
