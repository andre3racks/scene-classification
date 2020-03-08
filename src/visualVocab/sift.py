import cv2
import numpy as np

# creates a list of lists of sift keypoints for the given input images

def build_sift_descriptors(examples):
    keypoint_list, descriptor_list = [], []

    sift = cv2.xfeatures2d.SIFT_create()
    for ex in examples:
        kp, descriptors = sift.detectAndCompute(ex,None)
        keypoint_list.append(kp)
        descriptor_list.append(descriptors)
        # print keypoints on image
        # img=cv2.drawKeypoints(ex, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite('sift_keypoints.jpg',img)

    descriptor_list = np.asarray(descriptor_list)
    descriptor_list = np.concatenate(descriptor_list, axis=0)
    # print(descriptor_list.shape)

    return keypoint_list, descriptor_list

# debugging purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    build_sift_descriptors(data)
