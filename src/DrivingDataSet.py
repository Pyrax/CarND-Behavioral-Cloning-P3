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


def merge_camera_images(source_df, cameras=['Center', 'Left', 'Right'], camera_steering_offsets=[.0, .2, -.2],
                        verbose=0):
    """
    Expands camera images which are columns of the source_df into a new data frame as rows with an offset for the
    steering angles.

    :param source_df: data frame with camera columns
    :param cameras: specify which cameras to use, others are ignored
    :param camera_steering_offsets: offset for each camera to apply to steering angle (in range between 0.0 and 1.0)
    :param verbose: whether to output information about the data frames for testing
    :return: a single data frame with a list of [image file, steering angle] composed of all specified cameras
    """
    camera_dfs = [
        source_df[[f'{cam} image', 'Steering']]
            .rename(columns={f'{cam} image': 'Image'}) \
        for cam in cameras
    ]
    for idx, offset in enumerate(camera_steering_offsets):
        camera_dfs[idx]['Steering'] += offset

    # Checking that the mean steering angles are in fact different by the offset:
    if verbose > 0:
        means = [df['Steering'].mean() for df in camera_dfs]
        print(f'mean steering angles by cameras:\n{cameras}\n{str(means)}\n')

    # Merge all data frames into one
    return pd.concat(camera_dfs)


def df_add_inverted_copies(df, value_column='Steering'):
    """
    Takes a data frame and inserts a copy with inverted values. Also adds a new column CopyID responsible for marking
    the number of copies of the original image for augmentation.
    CopyID = 0 will be original image while CopyID = 1 will be flipped image.

    :param df: source data frame to invert values from
    :param value_column: name of column which contains the values which should be inverted
    :return: new data frame with original values and new rows with inverted values
    """
    tmp_df = df.copy()
    tmp_df['CopyID'] = 0

    copy_df = tmp_df.copy()
    copy_df['CopyID'] = 1
    copy_df[value_column] *= -1

    # Time to combine both data frames and reset index so that the data frame is re-indexed and eliminate duplicate
    # indices.
    return pd.concat([tmp_df, copy_df]).reset_index(drop=True)


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
    """
    Randomly perform several augmentation techniques on given data.
    Includes brightness, salt & pepper noise and image rotation.

    :param image: original image to augment
    :param steering: original steering angle which might be corrected
        if augmentation includes shifting for example
    :return: augmented image, corrected steering angle
    """
    # Brightness
    should_adjust_brightness = random.choice([False, True])
    if should_adjust_brightness:
        # only choose from a list of precomputed gamma tables for reduced computation time
        random_gamma = random.randint(0, len(precomputed_gammas) - 1)
        image = adjust_brightness(image, precomputed_lut=precomputed_gammas[random_gamma])

    # Salt and pepper noise
    should_add_noise = random.choice([False, True])
    if should_add_noise:
        random_noise = random.uniform(.2e-2, 5e-2)
        image = add_salt_pepper_noise(image, random_noise)

    # Rotation
    rotation_range = 10.0
    random_rotation = random.uniform(-rotation_range, rotation_range)
    image = rotate_image(image, random_rotation)
    return image, steering


def add_rotation(image, steering_angle, rotation=0.0):
    # Correct steering angle for rotation
    # Steering angles from simulator are between 0.0 and 1.0 where 1.0 equals 25°:
    adjusted_steering_angle = steering_angle + rotation / 25.0
    rotated_image = rotate_image(image, rotation)
    return rotated_image, np.clip(adjusted_steering_angle, None, 1.0)  # do not exceed steering limit of 1.0


def rotate_image(image, rotation=0.0):
    """
    Rotate an image by a given degree.

    :param image: original image to rotate
    :param rotation: rotation in degrees
    :return: rotated image
    """
    rows, cols, _ = image.shape
    center_coords = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    rot_m = cv2.getRotationMatrix2D(center_coords, rotation, 1)
    rotated_image = cv2.warpAffine(image, rot_m, (cols, rows))
    return rotated_image


def add_salt_pepper_noise(image, amount=0.0):
    """
    Add salt and pepper noise to an image.
    For RGB images it will add noise to all channels independently which
    means noise is colored also.
    See: https://stackoverflow.com/a/30624520

    :param image: original image
    :param amount: number of noisy pixels
    :return: noisy image
    """
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
    """
    Compute a lookup table for gamma correction for a given gamma factor.

    :param gamma: gamma correction factor
    :return: lookup table for gamma correction
    """
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return lut


gammas = np.linspace(0.5, 2.0, 20)
precomputed_gammas = [compute_gamma_lut(gamma) for gamma in gammas]
