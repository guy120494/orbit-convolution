from typing import Tuple, Any

import numpy as np
import tensorflow as tf


def get_mnist_data() -> Tuple[Any, Any, Any, Any]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # x_train = x_train[:640]
    # y_train = y_train[:640]

    x_train = x_train.astype(np.float32)
    x_train = np.reshape(x_train, (*x_train.shape, 1))

    x_test = x_test.astype(np.float32)
    x_test = np.reshape(x_test, (*x_test.shape, 1))

    # x_test = x_test[:640]
    # y_test = y_test[:640]

    return x_train / 255.0, y_train, x_test / 255.0, y_test


def get_cifar_data() -> Tuple[Any, Any, Any, Any]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # x_train = x_train[:640]
    # y_train = y_train[:640]

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # x_test = x_test[:640]
    # y_test = y_test[:640]

    return x_train / 255.0, y_train, x_test / 255.0, y_test


def get_datasets(name_of_dataset: str = "mnist"):
    if name_of_dataset == "cifar10":
        x_train, y_train, x_test, y_test = get_cifar_data()
    else:
        x_train, y_train, x_test, y_test = get_mnist_data()
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_set = train_set.shuffle(buffer_size=1024).batch(64)
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    return train_set, test_set
