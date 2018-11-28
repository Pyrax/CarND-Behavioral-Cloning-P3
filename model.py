import os
import argparse

from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, Cropping2D, Lambda, Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Nadam

from src import get_driving_data, steering_image_batch_generator, ResizeImages, merge_camera_images, \
    df_add_inverted_copies

CAM_OFFSET = .25
BATCH_SIZE = 32

LEFT_CROP = 30
RIGHT_CROP = 30
BOTTOM_CROP = 25
TOP_CROP = 65

ORIGINAL_IMAGE_HEIGHT = 160
ORIGINAL_IMAGE_WIDTH = 320

MODEL_IMAGE_HEIGHT = 66
MODEL_IMAGE_WIDTH = 200


def create_model():
    input_layer = Input(shape=(ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, 3), name='input_image')

    # Crop the input image first, then normalize it.
    x = Cropping2D(cropping=((TOP_CROP, BOTTOM_CROP), (LEFT_CROP, RIGHT_CROP)), name='image_cropping')(input_layer)
    x = ResizeImages((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), name='image_resize')(x)
    x = Lambda(lambda n: n / 127.5 - 1.0, name='image_normalization')(x)

    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name='conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', name='conv2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', name='conv3')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv4')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv5')(x)
    x = BatchNormalization()(x)
    x = Flatten(name='flatten')(x)
    x = Dense(100, activation='relu', name='dense1')(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu', name='dense2')(x)
    x = BatchNormalization()(x)
    x = Dense(10, activation='relu', name='dense3')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='tanh', name='output_angle')(x)

    model = Model(inputs=input_layer, outputs=x)
    opt = Nadam(lr=.001)  # lower learning rate as training otherwise gets stuck at higher loss
    model.compile(loss='mse', optimizer=opt)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model for autonomous driving')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to data directory. This should be the directory where driving_log.csv can be found.'
    )
    parser.add_argument(
        'model_file',
        type=str,
        nargs='?',
        default='PilotNet_model.h5',
        help='Path to file in which model should be saved. Should end with .h5'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Maximum number of epochs to train (early stopping might still be applied)'
    )
    args = parser.parse_args()

    driving_df = get_driving_data(os.path.join(args.data_path, 'driving_log.csv'))
    merged_df = merge_camera_images(driving_df, cameras=['Center', 'Left', 'Right'],
                                    camera_steering_offsets=[0.0, CAM_OFFSET, -CAM_OFFSET])
    full_df = df_add_inverted_copies(merged_df)

    train_set, validation_set = train_test_split(full_df.values, test_size=0.2)

    train_generator = steering_image_batch_generator(args.data_path, train_set, augment_data=True)
    validation_generator = steering_image_batch_generator(args.data_path, validation_set)

    epoch_steps_train = len(train_set) // BATCH_SIZE
    epoch_steps_validation = len(validation_set) // BATCH_SIZE

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(args.model_file, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

    PilotNet = create_model()
    PilotNet.fit_generator(train_generator,
                           validation_data=validation_generator,
                           epochs=args.epochs,
                           steps_per_epoch=epoch_steps_train,
                           validation_steps=epoch_steps_validation,
                           callbacks=[checkpoint, early])
