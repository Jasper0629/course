import numpy as np
import struct
import os
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import *
def load_img(load_path):

    with open(load_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num,-1)
    return img

def load_label(load_path):

    with open(load_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label
def load_minst():

    train_img = load_img('minst/train-images-idx3-ubyte')
    train_label = load_label('minst/train-labels-idx1-ubyte')
    test_img = load_img('minst/t10k-images-idx3-ubyte')
    test_label = load_label('minst/t10k-labels-idx1-ubyte')

    return train_img, train_label, test_img, test_label


if "__main__" == "__main__":
    train_img, train_label, test_img, test_label = load_minst()    
      
    assert train_img.shape[0] == train_label.shape[0], "train_img.shape[0] != train_label.shape[0]"
    assert test_img.shape[0] == test_label.shape[0], "test_img.shape[0] != test_label.shape[0]"

    print(train_img.shape)
    clf = SVC(C=1.0, kernel='poly', gamma=0.1)

    clf.fit(train_img, train_label)
    y_train_pred = clf.predict(train_img)
    y_test_pred = clf.predict(test_img)
    train_acc = accuracy_score(train_label, y_train_pred)
    test_acc = accuracy_score(test_label, y_test_pred)
