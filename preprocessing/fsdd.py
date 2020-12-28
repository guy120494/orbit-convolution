import os
import random
from shutil import copyfile

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


def split_train_test(source):
    dest_path = os.path.abspath(source)
    file_names = [f for f in os.listdir(source) if f.endswith('png')]
    for filename in file_names:
        first_split = filename.rsplit("_", 1)[1]
        second_split = first_split.rsplit(".", 1)[0]
        if int(second_split) <= 4:
            copyfile(source + "/" + filename, f"{dest_path}/test" + "/" + filename)
        else:
            copyfile(source + "/" + filename, f"{dest_path}/train" + "/" + filename)


def get_spectrograms(data_dir=None):
    """
    Args:
        data_dir (string): Path to the directory containing the spectrograms.
    Returns:
        (spectrograms, labels): a tuple of containing lists of spectrograms images(as numpy arrays) and their corresponding labels as strings
    """
    spectrograms = []
    labels = []

    if data_dir is None:
        data_dir = os.path.dirname(__file__) + '/../spectrograms'
        print(data_dir)

    file_paths = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and '.png' in f]

    if len(file_paths) == 0:
        raise Exception(
            'There are no files in the spectrogram directory. Make sure to run the spectrogram.py before calling this function.')

    for file_name in file_paths:
        label = int(file_name[0])
        label = keras.utils.to_categorical(label, num_classes=10)
        spectrogram = get_spectrogram_as_numpy(data_dir, file_name)
        spectrograms.append(spectrogram)
        labels.append(label)

    data_set = tf.data.Dataset.from_tensor_slices((spectrograms, labels))
    data_set = data_set.shuffle(buffer_size=2800).batch(128)
    return data_set


def get_spectrogram_as_numpy(data_dir, file_name):
    spectrogram = Image.open(data_dir + '/' + file_name)
    # spectrogram = spectrogram.convert('RGB')
    spectrogram = np.asarray(spectrogram)
    spectrogram = spectrogram / 255
    # spectrogram = translate_in_y_axis(spectrogram)
    # spectrogram = mild_translate_in_y_axis(spectrogram)
    return spectrogram


def translate_in_y_axis(img: np.ndarray):
    shift_number = random.randint(0, img.shape[1])
    return tf.roll(img, shift=shift_number, axis=1)


def mild_translate_in_y_axis(img: np.ndarray):
    shift_number = random.randint(0, 20)
    return tf.roll(img, shift=shift_number, axis=1)
