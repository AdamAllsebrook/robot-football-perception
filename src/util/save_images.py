"""
Helper for saving images into the dataset
"""
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path[-5:] == '/util':
    dir_path = dir_path[:-5]

class ImageSaver:

    def __init__(self):
        # create directories to store images in
        if not os.path.isdir(dir_path + '/classifier/dataset'):
            os.mkdir(dir_path + '/classifier/dataset')
        if not os.path.isdir(dir_path + '/classifier/dataset/resized'):
            os.mkdir(dir_path + '/classifier/dataset/resized')

        # load existing dataset image names
        # (images are named as 'ID.png' e.g. '1.png')
        self.existing_files = os.listdir(dir_path + '/classifier/dataset/resized')
        # remove .png file extension
        self.existing_files = list(map(lambda x: int(x[:-4]), self.existing_files))
        self.id = 0

    def save_image(self, image, size):
        # get the next lowest id that has not been used yet
        while self.id in self.existing_files:
            self.id += 1
        
        # resize and save the image
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dir_path + '/classifier/dataset/resized/%d.png' % self.id, resized)
        self.existing_files.append(self.id)
