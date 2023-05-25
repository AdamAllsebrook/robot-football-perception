"""
Train a classifier using src/classifier/train.py

python3 train_classifier.py [svm|adaboost]

(structured this way for imports)
"""

import sys

from classifier.train import train_classifier
from classifier.test import test

if __name__ == '__main__':
    if len(sys.argv) > 2:
        test(sys.argv[1])
    else:
        model = sys.argv[1]
        train_classifier(model)