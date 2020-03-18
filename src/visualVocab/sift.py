import cv2
import numpy as np

# sifts one image
def sifty(image):

    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptor = sift.detectAndCompute(image,None)

    if kp is None or descriptor is None:
        return None, None

    else:
        return kp, descriptor

# debugging purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    build_sift_descriptors(data)
