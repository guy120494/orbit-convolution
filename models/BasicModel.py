from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.python.keras.models import Model


class BasicModel(Model):

    def __init__(self):
        super(BasicModel, self).__init__()
        self.first_cnn_layer = None
        self.second_cnn_layer = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.max_pooling = MaxPooling2D(pool_size=(2, 2))
        self.first_dropout = Dropout(0.25)
        self.flatten = Flatten()
        self.first_dense = Dense(128, activation='relu')
        self.second_dropout = Dropout(0.5)
        self.dense = Dense(10, activation='softmax')

    def build(self, input_shape):
        self.first_cnn_layer = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.first_cnn_layer(inputs)
        x = self.second_cnn_layer(x)
        x = self.max_pooling(x)
        x = self.first_dropout(x, training=training)
        x = self.flatten(x)
        return self.dense(x)

    def get_config(self):
        pass
