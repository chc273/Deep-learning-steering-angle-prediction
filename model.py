# coding: utf-8

import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, Flatten, Lambda
from keras.preprocessing.image import load_img
import numpy as np
import cv2

crop_y = [60, 140]
RESIZE = (64, 64)


def get_one_image(image_file):
    """
    Read in one image file

    :param image_file: (str) full path of an image
    :return: (numpy.ndarray) image ndarray
    """

    image = np.asarray(load_img(image_file.strip()))
    return image


def crop_image(image, crop_y):
    """
    Crop the image with crop_y range

    :param image: (numpy.ndarray) image ndarray
    :param crop_y: (list) two item list or tuple providing the cropping range
    :return: (numpy array) cropped image ndarray
    """
    return image[crop_y[0]:crop_y[1], :, :]


def resize_image(image, RESIZE):
    """
    resize the image

    :param image: (numpy.ndarray) image ndarray
    :param RESIZE: (list) image size, length 2 tuple or list
    :return: (numpy array) image ndarray after resizing
    """
    image = cv2.resize(image, RESIZE, cv2.INTER_AREA) 
    return image


def get_one_image_crop(image_file, crop_y):
    """
    Buddle of two functions

    :param image_file: (numpy.ndarray) image file path
    :param crop_y: (list) crop range
    :return: (numpy array) cropped image
    """
    return crop_image(get_one_image(image_file), crop_y)


def shift_image(image, steering_angle, magx=120, magy=40):
    """
    Shift the image both horizontally and vertically, correspondingly
    the steering angles change. If the image is shifted to the left, we expect
    the new steering angle would be greater. On the other hand, if the image is
    shifted to the right, the angle should be smaller. Here I am using the change
    factor of 0.004/px.

    :param image: (numpy.ndarray) image ndarray
    :param steering_angle: (float) steering angle
    :param magx: (float) magnitude of x direction shift
    :param magy: (float) magnitude of y direction shift
    :return: (tuple) the shifted image, new_steering angle
    """
    rows, cols, _ = image.shape
    tx = magx*(np.random.rand(1) - 0.5)
    ty = magy*(np.random.rand(1) - 0.5)
    image = cv2.warpAffine(image, np.float32([[1, 0, tx], [0, 1, ty]]), (cols, rows))
    steering_angle = steering_angle + 0.004*(+tx) + 0.004 * ty * np.sign(steering_angle)
    return image, steering_angle


def flip_image(image, steering_angle):
    """
    Flip the image horizontally and change the steering angle sign

    :param image: (numpy.ndarray) image ndarray
    :param steering_angle: (float) steering angle
    :return: (tuple) flipped image ndarray, new_steering_angle
    """
    return image[:, ::-1, :], -steering_angle


def change_brightness(image, steering_angle):
    """
    Change the brightness of the road without changing steering angle

    :param image: (numpy.ndarray) image ndarray
    :param steering_angle: (float) steering angle
    :return: (tuple), image and new steering angle

    """
    scale = int((np.random.rand(1) - 0.8)*255)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:, :, 2] = cv2.add(image[:, :, 2], scale)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image, steering_angle


def random_rotation(image, steering_angle, mag=20):
    """
    Randomly rotate the image and change the steering angle correspondingly

    :param image: (numpy.ndarray) image ndarray
    :param steering_angle: (float) steering angle
    :param mag: (float) magnitude of rotation
    :return: (tuple) rotated image, new steering angle
    """
    rotate_angle = mag * (np.random.rand(1) - 0.5)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotate_angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    steering_angle -= rotate_angle / 40.0
    return image, steering_angle


def shear_image(image, steering_angle, mag=60):
    """
    Shear the image and change the steering angle

    :param image: (numpy.ndarray) image ndarray
    :param steering_angle: (float) steering angle
    :param mag: (float) magnitude of shearing
    :return: (tuple) sheared image, new steering angle
    """
    rows, cols, _ = image.shape
    shear_angle = (np.random.rand(1)-0.5)*mag
    pts1 = np.float32([[0, 80], [320, 80], [shear_angle+100, 0], [shear_angle + 320-100, 0]])
    pts2 = np.float32([[0, 80], [320, 80], [100, 0], [220, 0]])
    m = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, m, (cols, rows))
    return image, steering_angle - shear_angle / 200. 
    
    
def image_generator(image_index, batch_size):
    """
    Generate images from driving_log.csv and the image file directory

    :param image_index: (pandas.DataFrame) image file name and steering angle
    :param batch_size: (int) batch size
    :return: (tuple) image array and steering angle
    """

    while True:
        
        random_index = np.random.permutation(image_index.index)[:batch_size]
        image_data = image_index.iloc[random_index]
        image_data.reset_index(drop=True, inplace=True)
        steering_angle = np.asarray(image_data['steering'])
        n_sub = image_data.shape[0]
        image_temp = np.zeros(shape=(n_sub, RESIZE[0], RESIZE[1], 3))
        
        for i, j in image_data.iterrows():
            # flip a coin and decide which camera will be used
            # For the center image, the steering angle does not change
            # For the left camera, I correct the angle by +1./25 degrees
            # For the right camera, the angle is corrected by -1/25 degrees
            column_index = np.random.randint(3)  # 0: center, 1: left, 2: right
            steering_factor = 0.15  # 180./np.pi*1./15.0/25.0

            image = get_one_image(j.iloc[column_index])
            image = crop_image(image, crop_y)
            if column_index == 0:
                dsteering = 0
            elif column_index == 1:
                dsteering = steering_factor
            else:
                dsteering = -steering_factor
            steering_angle[i] += dsteering

            # Data augmentation
            is_shift = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
            if is_shift:
                image, new_angle = shift_image(image, steering_angle[i])
                steering_angle[i] = new_angle

            # random flip
            is_flip = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
            if is_flip:
                image, new_angle = flip_image(image, steering_angle[i])
                steering_angle[i] = new_angle
                
            # random brightness
            is_bright_change = np.random.choice([0, 1], size=1, p=[0.2, 0.8])
            if is_bright_change:
                image, _ = change_brightness(image, steering_angle[i])

            # random rotation
            is_rotate = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
            if is_rotate:
                image, new_angle = random_rotation(image, steering_angle[i])
                steering_angle[i] = new_angle

            # random shear
            is_shear = np.random.choice([0, 1], size=1, p=[0.7, 0.3])
            if is_shear:
                image, new_angle = shear_image(image, steering_angle[i])
                steering_angle[i] = new_angle

            image = resize_image(image, RESIZE)
            image_temp[i, :, :, :] = image

        yield image_temp, steering_angle


def get_validation(image_index):
    """
    Get the center image data as the validation.

    :param image_index: (pandas.DataFrame) image names and ther
    :return:
    """
    N = image_index.shape[0]
    X = np.zeros(shape=(N, 64, 64, 3))
    Y = image_index['steering']
    for i in range(N):
        X[i, :, :, :] = resize_image(get_one_image_crop(image_index.loc[i, 'center'], crop_y), RESIZE)
    return X, np.array(Y)


image_index = pd.read_csv('driving_log.csv')


def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape= (64, 64, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model


batch_size = 200
nb_epoch = 10
N = 40000 

model = nvidia()
print(model.summary())
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse")
generator = image_generator(image_index, batch_size)
model.fit_generator(generator, samples_per_epoch=N,  nb_epoch=nb_epoch,
                    verbose=1, validation_data=get_validation(image_index))

json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights('model.h5')

