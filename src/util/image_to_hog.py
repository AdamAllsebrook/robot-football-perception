"""
Convert an image to a histogram of oriented gradients feature vector
"""
from skimage.feature import hog


# use the same parameters for model training and during classification
def image_to_hog(image):
    return hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, feature_vector=True), None

def get_hog_feature(image):
    fd, _ = image_to_hog(image)
    return fd
