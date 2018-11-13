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


def parse_data_row(data_path, row):
    center_image_name, left_image_name, right_image_name, steering = row[0:4]

    center_image = cv2.imread(os.path.join(data_path, center_image_name))
    return center_image, steering


def steering_image_batch_generator(data_path, samples, batch_size=32):
    """
    Generates shuffled batches of image data paired with ground-truth labels by reading
    image files from a list of file names.

    :param data_path: directory where to find the image files
    :param samples: list of image names and steering angles from driving_log.csv
    :param batch_size: number of samples each batch should have
    :return: list of [images, labels] of next batch
    """
    num_samples = len(samples)
    # endless loop for batch generation
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image, steering = parse_data_row(data_path, batch_sample)

                images.append(center_image)
                angles.append(steering)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)
