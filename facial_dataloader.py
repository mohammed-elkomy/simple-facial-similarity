"""
Data:git clone https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch.git
path:Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/faces/training
path:Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/faces/testing
0 same ,1 different
"""

import os
import random

import cv2
import numpy as np
from imgaug import augmenters as iaa
from tensorflow import keras

seq = iaa.Sequential([

    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Sometimes(0.9, iaa.Crop(percent=((0.0, 0.15), (0.0, 0.15), (0.0, 0.15), (0.0, 0.15)))),  # random crops top,right,bottom,left
    # some noise
    iaa.Sometimes(0.5, [iaa.GaussianBlur(sigma=(0, 0.25)), iaa.Sharpen(alpha=(0.0, .1), lightness=(0.5, 1.25)), iaa.Emboss(alpha=(0.0, 1.0), strength=(0.05, 0.1))]),
    iaa.Sometimes(.7, iaa.Add((-10, 10), per_channel=0)),
    iaa.Scale({"height": 112, "width": 92})

], random_order=True)  # apply augmenters in random order


class FacialSequence(keras.utils.Sequence):
    def __init__(self, data_root_path, batch_size, augmenter=seq):
        """get data structure to load data"""
        # list of (images paths,image root path index(person))
        self.data_source = []

        person_id = 0
        for root, dirs, files in os.walk(data_root_path):  # this will do the dfs for you so you can get every single file and directory
            # this is a person
            if files and not dirs:
                for file in files:
                    self.data_source.append((os.path.join(root, file), person_id))
                person_id += 1

        print("Dataset size", len(self.data_source))
        # pprint(self.data_source)
        self.batch_size = batch_size

        self.augmenter = augmenter

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # ceiling div

    def get_actual_length(self):
        """Denotes the total data_to_load of samples"""
        return len(self.data_source)

    def get_pair_of_images(self, is_required_different):  # 0 same ,1 different
        while True:
            first, second = random.choice(self.data_source), random.choice(self.data_source)
            first_label, second_label = first[1], second[1]
            is_samples_different = first_label != second_label
            if is_required_different == is_samples_different:
                # print(is_required_different, first[0], second[0])
                return cv2.cvtColor(cv2.imread(first[0]), cv2.COLOR_BGR2GRAY), \
                       cv2.cvtColor(cv2.imread(second[0]), cv2.COLOR_BGR2GRAY)

    def __getitem__(self, _):
        """Gets one batch"""
        sister1_batch = []
        sister2_batch = []
        labels_batch = np.zeros((self.batch_size,)).astype(np.float32)

        for sample in range(self.batch_size):
            label = random.randint(0, 1)  # 0 same ,1 different.. uniform distribution of course
            sister1, sister2 = self.get_pair_of_images(label)
            sister1_batch.append(sister1)
            sister2_batch.append(sister2)
            labels_batch[sample] = label

        # [batch,height,width,1],[batch,height,width,1],[batch,]
        return [
                   np.expand_dims(np.array(self.augmenter.augment_images(sister1_batch), dtype=np.float32) / 255.0, axis=-1),
                   np.expand_dims(np.array(self.augmenter.augment_images(sister2_batch), dtype=np.float32) / 255.0, axis=-1),
               ], np.reshape(labels_batch,(-1))

    def get_data_source(self):
        return list(map(lambda item: item[0], self.data_source))


if __name__ == '__main__':

    train_loader = FacialSequence(data_root_path="./data/faces/training",
                                  batch_size=1)

    test_loader = FacialSequence(data_root_path="./data/faces/testing",
                                 batch_size=1)
    loader = test_loader
    sss = 0
    for i in range(1000):
        sample = loader[i]
        s1, s2 = sample[0]
        labels = sample[1]
        s1, s2, labels = s1[0], s2[0], labels[0]

        # # VIEW TEST
        # cv2.namedWindow("s1".format(i), cv2.WINDOW_NORMAL)
        # cv2.moveWindow("s1".format(i), 0, 0)
        # cv2.imshow("s1".format(i), s1)
        #
        # cv2.namedWindow("s2".format(i), cv2.WINDOW_NORMAL)
        # cv2.moveWindow("s2".format(i), 400, 400)
        # cv2.imshow("s2".format(i), s2)
        #
        # print(labels)
        # cv2.waitKey()

        # BALANCE TEST
        sss += labels
        if i % 100 == 0:
            print(i)

    print(sss, 10000 / 2)
