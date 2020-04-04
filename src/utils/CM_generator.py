import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import time

def plot_confusion_matrix(cm, title="Confusion Matrix", save=True):
    plt.figure()
    classes = np.arange(cm.shape[0])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save:
        plt.savefig('confusion_matrix.png'.format(title), bbox_inches='tight')
        return


