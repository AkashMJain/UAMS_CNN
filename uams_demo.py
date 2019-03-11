from statistics import mode

import cv2
from keras.models import load_model
import numpy as np


class UAMS_CNN(object):
    """docstring for UAMS_CNN"""

    def __init__(self):
        super(UAMS_CNN, self).__init__()
        detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
        emotion_model_path = 'trained_models/your_model_name'
        emotion_labels = get_labels('uams_dataset')

    # create your own dataset and pass it to here
    def get_labels(dataset_name):
        if dataset_name == 'uams_dataset':
            return {0: 'attentive', 1: 'un-attentive'}

        else:
            raise Exception('Invalid dataset name')
