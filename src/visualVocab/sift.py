import cv2
import numpy as np

# densely sifts one image
def sifty(image):

    sift = cv2.xfeatures2d.SIFT_create()

    step_size = 5
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, image.shape[0], step_size) 
                                        for x in range(0, image.shape[1], step_size)]
    # kp, descriptor = sift.detectAndCompute(image,None)

    keypoints, descriptors = sift.compute(image, kp)

    # if kp is None or descriptor is None:
    if keypoints is None or descriptors is None:
        return None, None

    else:
        return keypoints, descriptors

# debugging purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    build_sift_descriptors(data)
