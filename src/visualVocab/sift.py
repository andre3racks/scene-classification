import cv2
import numpy as np

# debugging purposes
# data = {}
# data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]

# creates a list of lists of sift keypoints for the given input images

def build_sift_descriptors(data):
    keypoint_list = []

    sift = cv2.xfeatures2d.SIFT_create()
    for ex in data['X_train']:
        kp = sift.detect(ex,None)
        keypoint_list.append(kp)
        # print keypoints on image
        img=cv2.drawKeypoints(ex, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('sift_keypoints.jpg',img)

    

    return keypoint_list

# debugging purposes
# build_sift_descriptors(data)