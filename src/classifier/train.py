"""
Train a classifier using the classified data

Use src/train_classifier.py
"""

import numpy as np
import cv2
import os
import time
import timeit

from joblib import dump
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from util.image_to_hog import get_hog_feature

dir_path = os.path.dirname(os.path.realpath(__file__))


def augment_images(images):
    """
    Augment a list of images by
        - flipping horizontally
        - rotating by -20, -10, 10, 20 degrees
    Increases size of dataset by 10x
    """
    augmented_images = []

    h, w = images[0].shape[:2]
    image_center = (int(w/2), int(h/2))
    # calculate matrix for each rotation
    rotation_matrices = [cv2.getRotationMatrix2D(image_center, angle, 1) for angle in [-20, -10, 10, 20]]

    for original in images:
        hor_flip = cv2.flip(original, 1)
        augmented_images += [original, hor_flip]
        # rotate original and flipped by -20, -10, 10, 20 degrees
        for image in [original, hor_flip]:
            for M in rotation_matrices:
                augmented_images.append(cv2.warpAffine(image, M, (w, h)))

    return augmented_images


def train_svm(samples, classes):
    """
    Train an SVM classifier
    """
    classifier = svm.SVC(kernel='rbf')
    classifier.fit(samples, classes)

    dump(classifier, dir_path + '/svm_classifier.gz')

    t = timeit.timeit(lambda: classifier.predict(np.array([samples[np.random.randint(samples.shape[0])]])), number=1000)
    print('time to classify 1000 images: %.5f' % t)

    print('SVM classifier successfully trained!')


def train_adaboost(samples, classes):
    """
    Train an adaboost classifier
    """
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=100
    )

    bdt.fit(samples, classes)

    dump(bdt, dir_path + '/adaboost_classifier.gz')

    t = timeit.timeit(lambda: bdt.predict(np.array([samples[np.random.randint(samples.shape[0])]])), number=1000)
    print('time to classify 1000 images: %.5f' % t)

    print('adaboost classifier successfully trained!')


def train_classifier(model):
    """
    Train either an SVM or AdaBoost classifier using the image in /dataset/
    Positive samples are augmented
    """
    if os.path.isdir(dir_path + '/dataset/noball') and os.path.isdir(dir_path + '/dataset/ball'):
        # load dataset images
        noball_files = os.listdir(dir_path + '/dataset/noball')
        ball_files = os.listdir(dir_path + '/dataset/ball')

        noball_images = list(map(lambda x: cv2.imread(
            dir_path + '/dataset/noball/' + x), noball_files))
        ball_images = list(map(lambda x: cv2.imread(
            dir_path + '/dataset/ball/' + x), ball_files))

        ball_images = augment_images(ball_images)

        print('%d ball images (augmented x10), %d noball images' % (len(ball_images), len(noball_images)))

        all_images = noball_images + ball_images
        classes = np.array([0] * len(noball_images) + [1] * len(ball_images))

        # calculate HOG feature for all samples
        all_hog_features = np.array(
            list(map(lambda x: get_hog_feature(x), all_images)))

        start_time = time.time()
        if model == 'svm':
            train_svm(all_hog_features, classes)

        elif model == 'adaboost':
            train_adaboost(all_hog_features, classes)
        print('took %.3f seconds' % (time.time() - start_time))
