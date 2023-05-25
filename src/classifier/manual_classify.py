#!/usr/bin/env python3

"""
Manually classify images as either being a ball or not
Controls:
    ball: a/j
    noball: d/l

    scroll left: left arrow
    scroll right: right arrow
    quit: q
"""

import os
import cv2
import numpy as np
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
unclassified_path = dir_path + '/dataset/resized'

stack = []

if len(sys.argv) > 1:
    prefix = sys.argv[1]
else:
    prefix = ''



def save_image(image, filepath, chosen_class):
    """
    Save an image in the given class.
    Images are named numerically
    """
    i = len(os.listdir(dir_path + '/dataset/' + prefix + '_' + chosen_class))
    new_filepath = dir_path + '/dataset/' + prefix + '_' + \
        chosen_class + '/' + str(i) + '.png'
    cv2.imwrite(new_filepath, image)
    os.remove(filepath)

    return (image, new_filepath)


def find_first_unclassified(dataset):
    """
    Find the first image in the dataset that has not been classified
    """
    for i, (_, filepath) in enumerate(dataset):
        if 'resized' in filepath:
            return i
    return len(dataset)


if __name__ == '__main__':
    if os.path.isdir(unclassified_path):

        # create directories if they do not exist
        if not os.path.isdir(dir_path + '/dataset/' + prefix + '_ball'):
            os.mkdir(dir_path + '/dataset/' + prefix + 'ball')
        if not os.path.isdir(dir_path + '/dataset/' + prefix + '_noball'):
            os.mkdir(dir_path + '/dataset/' + prefix + 'noball')

        # load images into a list of (image, filepath)
        files = os.listdir(unclassified_path)
        dataset = list(map(lambda filename: (
            cv2.imread(unclassified_path + '/' + filename),
            unclassified_path + '/' + filename
        ), files))

        # iterate through each image in the dataset, displaying and classifying it
        i = 0
        while i < len(dataset):
            (image, filepath) = dataset[i]
            # resize image to be displayed
            # enlarged = cv2.resize(
            #     image, (256, 256), interpolation=cv2.INTER_LINEAR)

            # get next images to display as well
            enlarged = [cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR) 
                for (img, _) in dataset[i:min(i+5, len(dataset))]]
            enlarged_combined = np.concatenate(enlarged, axis=1)

            # get the class if the image has already been classified
            image_class = ''
            if 'noball/' in filepath:
                image_class = '✖'
            elif 'ball/' in filepath:
                image_class = '✓'

            # display the image
            cv2.imshow('%s %d/%d  ✓a ✖d' %
                       (image_class, i+1, len(dataset)), enlarged_combined)

            # wait for user input
            k = cv2.waitKey(0)
            if k in [97, 100, 106, 108]:
                # ball
                if k == 97 or k == 106:  # a, j
                    # do not allow the image to be reclassified as the same class
                    if 'ball/' not in filepath:
                        del dataset[i]
                        data = save_image(image, filepath, 'ball')
                        dataset.insert(i, data)
                # noball
                elif k == 100 or k == 108:  # d, l
                    if 'noball/' not in filepath:
                        del dataset[i]
                        data = save_image(image, filepath, 'noball')
                        dataset.insert(i, data)
                i = find_first_unclassified(dataset)

            # scroll left
            elif k == 81:  # left arrow
                i = max(i-1, 0)

            # scroll right
            elif k == 83:  # right arrow
                i += 1

            # quit
            elif k == 113:  # q
                break

            cv2.destroyAllWindows()
