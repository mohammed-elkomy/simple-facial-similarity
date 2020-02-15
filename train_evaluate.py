"""
  Created by mohammed-alaa

!git clone https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch.git
!pip install imgaug
%matplotlib inline
"""
import multiprocessing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Conv2D, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from facial_dataloader import FacialSequence

# #######################################################################
sister = Sequential()
# input: 112x92 images with 1 channels -> (112, 92, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
sister.add(Conv2D(4, (3, 3), activation='relu', input_shape=(112, 92, 1), padding="same"))
sister.add(BatchNormalization(axis=-1))  # color space

sister.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
sister.add(BatchNormalization(axis=-1))  # color space

sister.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
sister.add(BatchNormalization(axis=-1))  # color space

sister.add(Flatten())
sister.add(Dense(64, activation='relu'))
sister.add(Dense(8))
sister.add(BatchNormalization(axis=-1))  # normalized euclidean space

sister1_input = Input(shape=(112, 92, 1))
sister2_input = Input(shape=(112, 92, 1))
sister1_output = sister(sister1_input)
sister2_output = sister(sister2_input)


def L2_distance_layer(siamese_outputs):
    _sister1_output, _sister2_output = siamese_outputs
    return K.sum(K.square(_sister1_output - _sister2_output), axis=1, keepdims=True)  # 0 to ~8 after normalization


# note that "output_shape" isn't necessary with the TensorFlow backend
l2_squared = Lambda(L2_distance_layer)([sister1_output, sister2_output])  # l2_squared =pow(l2,2)

siamese = Model(inputs=[sister1_input, sister2_input], outputs=l2_squared)


def siamese_loss(is_different, _l2_squared):  # 0 same ,1 different
    return (1 - is_different) * _l2_squared + is_different * K.pow(tf.maximum(0.0, margin - K.sqrt(_l2_squared)), 2.0)


batch_size = 256
margin = 1.5
epochs = 25
train_enqueuer = FacialSequence(batch_size=batch_size, data_root_path="./data/faces/training")
test_enqueuer = FacialSequence(batch_size=batch_size, data_root_path="./data/faces/testing")
siamese.compile(loss=siamese_loss, optimizer=Adam(lr=10e-6))
siamese.fit_generator(generator=train_enqueuer, steps_per_epoch=40,  # (370 + batch_size - 1) // batch_size,
                      validation_data=test_enqueuer, validation_steps=4,  # (30 + batch_size - 1) // batch_size,
                      epochs=epochs, use_multiprocessing=True, workers=multiprocessing.cpu_count()
                      )
# ########################################################################################
# sister = Sequential()
# # input: 112x92 images with 1 channels -> (112, 92, 1) tensors.
# # this applies 32 convolution filters of size 3x3 each.
# sister.add(Conv2D(4, (3, 3), activation='relu', input_shape=(112, 92, 1), padding="same"))
# sister.add(BatchNormalization(axis=-1))  # color space
#
# sister.add(Conv2D(8, (3, 3), activation='relu', padding="same"))
# sister.add(BatchNormalization(axis=-1))  # color space
#
# sister.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
# sister.add(BatchNormalization(axis=-1))  # color space
#
# sister.add(Flatten())
# sister.add(Dense(64, activation='relu'))
# sister.add(Dense(8))
#
# sister1_input = Input(shape=(112, 92, 1))
# sister2_input = Input(shape=(112, 92, 1))
# sister1_output = sister(sister1_input)
# sister2_output = sister(sister2_input)
#
#
# def cosine_distance_layer(siamese_outputs):
#     _sister1_output, _sister2_output = siamese_outputs
#     _sister1_output = _sister1_output / (K.sqrt(K.sum(K.pow(_sister1_output, 2), axis=-1, keepdims=True)) + 1e-11)  # l2 normalization
#     _sister2_output = _sister2_output / (K.sqrt(K.sum(K.pow(_sister2_output, 2), axis=-1, keepdims=True)) + 1e-11)  # l2 normalization
#
#     return K.sum(_sister1_output * _sister2_output, axis=1, keepdims=True)  # element-wise multiplication..sum
#
#
# # note that "output_shape" isn't necessary with the TensorFlow backend
# cosine_distance = Lambda(cosine_distance_layer)([sister1_output, sister2_output])  # l2_squared =pow(l2,2)
#
# siamese = Model(inputs=[sister1_input, sister2_input], outputs=cosine_distance)
#
#
# def siamese_loss(is_different, _cosine_distance):  # 0 same ,1 different
#     return (1 - is_different) * .25 * K.pow((1 - _cosine_distance), 2) + \
#            is_different * tf.where(_cosine_distance < margin, K.pow(_cosine_distance, 2), K.zeros_like(_cosine_distance))
#
# batch_size = 256
# margin = .9
# epochs = 25
# train_enqueuer = FacialSequence(batch_size=batch_size, data_root_path="./data/faces/training")
# test_enqueuer = FacialSequence(batch_size=batch_size, data_root_path="./data/faces/testing")
# siamese.compile(loss=siamese_loss, optimizer=Adam(lr=10e-6))
# siamese.fit_generator(generator=train_enqueuer, steps_per_epoch=40,  # (370 + batch_size - 1) // batch_size,
#                       validation_data=test_enqueuer, validation_steps=4,  # (30 + batch_size - 1) // batch_size,
#                       epochs=epochs, use_multiprocessing=True, workers=multiprocessing.cpu_count()
#                       )
# #############################################################################################
# evaluation


# evaluation


query_image = './data/faces/testing/s7/5.pgm'


# preview raw data
def preview_predictions(predictions):
    # 3 channels
    fig, axeslist = plt.subplots(ncols=7, nrows=6, figsize=(15, 15))

    for i, (path, score) in enumerate(predictions):
        axeslist.ravel()[i].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), cmap="gray")
        axeslist.ravel()[i].set_title(score)
        axeslist.ravel()[i].set_axis_off()


# that_images = sorted(list(filter(lambda item: "y" not in item, test_enqueuer.get_data_source())))
that_images = sorted(test_enqueuer.get_data_source())

that_batch = []
for that_image in that_images:
    that_batch.append(cv2.resize(cv2.cvtColor(cv2.imread(that_image), cv2.COLOR_BGR2GRAY), (92, 112)))
that_batch = np.expand_dims(np.array(that_batch), axis=-1).astype(np.float32) / 255.0
print(that_batch.shape)

this_image = np.expand_dims(cv2.resize(cv2.cvtColor(cv2.imread(query_image), cv2.COLOR_BGR2GRAY), (92, 112)), axis=0)
this_batch = np.expand_dims(np.repeat(this_image, len(that_images), axis=0), axis=-1).astype(np.float32) / 255.0
print(this_batch.shape)

query_predictions = sorted(list(zip(that_images, siamese.predict([this_batch, that_batch])[:, 0].tolist())), key=lambda item: item[1])
preview_predictions(query_predictions)
