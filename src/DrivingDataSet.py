import os
import sklearn
import cv2
import numpy as np
import pandas as pd
from random import shuffle


def get_driving_data(log_file_csv):
    """
    Reads data from a csv driving log.

    :param log_file_csv: path to .csv driving_log
    :return: pandas data frame of image file names and corresponding steering angle
    """
    return pd.read_csv(log_file_csv,
                       names=['Center image', 'Left image', 'Right image', 'Steering', 'Throttle', 'Brake',
                              'Speed'],
                       usecols=['Center image', 'Left image', 'Right image', 'Steering'])


def steering_image_batch_generator(data_path, samples, batch_size=32):
    """
    Generates shuffled batches of image data paired with ground-truth labels by reading
    image files from a list of file names.

    :param data_path: directory where to find the image files
    :param samples: list of image names and steering angles
    :param batch_size: number of samples each batch should have
    :return: list of [images, labels] of next batch
    """
    num_samples = len(samples)
    # endless loop for batch generation
    while 1:
        # Shuffle data at the beginning of each epoch
        shuffled_samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuffled_samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = batch_sample[0].strip()  # take care of whitespaces in file paths
                steering = batch_sample[1]
                image = cv2.imread(os.path.join(data_path, image_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                images.append(image)
                angles.append(steering)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)
