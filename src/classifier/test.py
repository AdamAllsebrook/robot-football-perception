import os
import cv2
import numpy as np
import sys

from joblib import load
from sklearn.metrics import confusion_matrix

from classifier.train import augment_images
from util.image_to_hog import get_hog_feature

dir_path = os.path.dirname(os.path.realpath(__file__))


def test(cls):
    if os.path.isdir(dir_path + '/dataset/test_noball') and os.path.isdir(dir_path + '/dataset/test_ball'):
        # load dataset images
        noball_files = os.listdir(dir_path + '/dataset/test_noball')
        ball_files = os.listdir(dir_path + '/dataset/test_ball')

        noball_images = list(map(lambda x: cv2.imread(
            dir_path + '/dataset/test_noball/' + x), noball_files))
        ball_images = list(map(lambda x: cv2.imread(
            dir_path + '/dataset/test_ball/' + x), ball_files))

        ball_images = augment_images(ball_images)

        print('%d ball images (augmented x10), %d noball images' % (len(ball_images), len(noball_images)))

        all_images = noball_images + ball_images
        classes = np.array([0] * len(noball_images) + [1] * len(ball_images))

        # calculate HOG feature for all samples
        all_hog_features = np.array(
            list(map(lambda x: get_hog_feature(x), all_images)))


        classifier = load(dir_path + '/' + cls + '_classifier.gz')

        predicted = np.array([
            classifier.predict(np.array([feature])) for feature in all_hog_features
        ])

        confusion = confusion_matrix(
            classes,
            predicted,
            labels=[0,1]
        )

        print(confusion)
