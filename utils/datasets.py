from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2


class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""

    def __init__(self, dataset_name='imdb',
                 dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size

        if self.dataset_path is not None:
            self.dataset_path = dataset_path

        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'

        elif self.dataset_name == 'uams-dataset':
            self.dataset_path = 'datasets/uams/uams-dataset.csv'

        else:
            raise Exception(
                'Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()

        elif self.dataset_name == 'uams-dataset':
            ground_truth_data = self._load_UAMSEmotion()
        return ground_truth_data

    def _load_UAMSEmotion(self):
        print("hello")
        print(self.dataset_path)
        data = pd.read_csv(self.dataset_path)
        print("i think you should reah here")
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split) * num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    elif dataset_name == 'uams-dataset':
        return {0: 'attentive', 1: 'un-attentive'}
    else:
        raise Exception('Invalid dataset name')
