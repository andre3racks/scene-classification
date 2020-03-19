import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import visualVocab

def directory_walk(rootDir):

    num_exs = 1
    data = {}

    for dirName, subdirList, fileList in os.walk(rootDir):

        # print("dir: {}".format(dirName))

        for file in fileList:
            # print("file {}".format(file))
            append_example(data, dirName, file, rootDir)
            num_exs += 1

            if num_exs % 10000 == 0:
                print("{} image read into data dictionary.".format(num_exs))
            None

    print("shuffling, ranging, and splitting data.")

    data = split(data)

    return data

# append sift descriptors of example or skip
def append_example(data, dirName, file, rootDir):

    img = cv2.imread(os.path.join(dirName, file), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image read failed in data_loader.")
        return

    # sift doesn't like normalized images
    # img = img.astype(np.float32) / 255
    # img -= img.mean()
    # img /= img.std()

    # print(img)
    
    kp, descriptor =  visualVocab.sift.sifty(img)

    if kp is None or descriptor is None:
        print("Sift failed in data loader.")
        return

    elif 'X_train' in data.keys():
        data['X_train'].append(descriptor)
        data['Y_train'].append(dirName)

    else:
        data['X_train'] = [descriptor]
        data['Y_train'] = [dirName]
        # print("init dict")
    

def split(data):
    # shuffle data
    data['X_train'], data['Y_train'] = shuffle(data['X_train'], data['Y_train'])

    # split
    data['X_train'], data['X_test'], data['Y_train'], data['Y_test'] = train_test_split(data['X_train'], data['Y_train'], test_size=0.2, random_state=1)

    return data