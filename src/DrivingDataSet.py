import os
import random
import sklearn
import cv2
import numpy as np
import pandas as pd


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


def steering_image_batch_generator(data_path, samples, batch_size=32, augment_data=False):
    """
    Generates shuffled batches of image data paired with ground-truth labels by reading
    image files from a list of file names.

    :param data_path: directory where to find the image files
    :param samples: list of image names and steering angles
    :param batch_size: number of samples each batch should have
    :param augment_data: whether to augment image data (should only be used for training)
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
                copy_id = batch_sample[2]

                image = cv2.imread(os.path.join(data_path, image_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if copy_id == 1:
                    image = np.fliplr(image)

                if augment_data:
                    image, steering = randomly_augment_data(image, steering)

                images.append(image)
                angles.append(steering)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


def randomly_augment_data(image, steering):
    # Brightness
    should_adjust_brightness = random.choice([False, True])
    # should_adjust_brightness = False
    if should_adjust_brightness:
        rand_gamma = random.randint(0, len(precomputed_gammas) - 1)
        image = adjust_brightness(image, precomputed_lut=precomputed_gammas[rand_gamma])
        # random_gamma = random.uniform(0.5, 2.0)
        # image = adjust_brightness(image, random_gamma)

    # Salt and pepper noise
    should_add_noise = random.choice([False, True])
    if should_add_noise:
        random_noise = random.uniform(1e-3, 9e-3)
        image = add_salt_pepper_noise(image, random_noise)

    # Rotation
    rotation_range = 10.0
    random_rotation = random.uniform(-rotation_range, rotation_range)
    image, _ = add_rotation(image, steering, random_rotation)
    return image, steering


def add_rotation(image, steering_angle, rotation=0.0):
    # Correct steering angle for rotation
    # Steering angles from simulator are between 0.0 and 1.0 where 1.0 equals 25°:
    adjusted_steering_angle = steering_angle + rotation / 25.0
    rotated_image = rotate_image(image, rotation)
    return rotated_image, np.clip(adjusted_steering_angle, None, 1.0)  # do not exceed steering limit of 1.0


def rotate_image(image, rotation=0.0):
    rows, cols, _ = image.shape
    center_coords = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    rot_m = cv2.getRotationMatrix2D(center_coords, rotation, 1)
    rotated_image = cv2.warpAffine(image, rot_m, (cols, rows))
    return rotated_image


def add_salt_pepper_noise(image, amount=0.0):
    # See: https://stackoverflow.com/a/30624520
    salt_vs_pepper = 0.5
    noisy_image = np.copy(image)

    # Salt
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255

    # Pepper
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    return noisy_image


def adjust_brightness(image, gamma=1.0, precomputed_lut=None):
    """
    Brightness adjustment using gamma correction.
    See: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

    :param image: original image to be adjusted
    :param gamma:
        gamma correction factor
        (for gamma < 1.0 dark regions will be brighter and for gamma > 1.0 dark regions will be darker)
    :param precomputed_lut:
        Use a precomputed lookup table for gamma values. Can be used to reduce execution time.
        gamma will be ignored if precomputed_lut is given.
    :return: copy of brightness adjusted image
    """
    if precomputed_lut is not None:
        return cv2.LUT(image, precomputed_lut)
    return cv2.LUT(image, compute_gamma_lut(gamma))


def compute_gamma_lut(gamma=1.0):
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return lut


gammas = np.linspace(0.5, 2.0, 20)
precomputed_gammas = [compute_gamma_lut(gamma) for gamma in gammas]
