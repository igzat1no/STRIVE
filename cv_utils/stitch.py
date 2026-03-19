import cv2
import numpy as np

def image_stitch_and_crop(image1, image2, camera_intrinsic):

    h, w = image1.shape[:2]

    def get_homography(theta):
        K = camera_intrinsic
        theta_rad = np.radians(theta)
        R = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])
        return K @ R @ np.linalg.inv(K)

    H1 = get_homography(-15)
    H2 = get_homography(15)

    warped1 = cv2.warpPerspective(image1, H1, (w, h))
    warped2 = cv2.warpPerspective(image2, H2, (w, h))

    merged = np.maximum(warped1, warped2)

    return merged


def combine_image(image1, image2, image3, camera_intrinsic):
    h, w = image1.shape[:2]

    def get_homography(theta):
        K = camera_intrinsic
        theta_rad = np.radians(theta)
        R = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])
        mat = K @ R @ np.linalg.inv(K)
        translate = np.array([
            [1, 0, w],
            [0, 1, 0],
            [0, 0, 1]
        ])
        mat = translate @ mat
        return mat

    H1 = get_homography(-30)
    H2 = get_homography(0)
    H3 = get_homography(30)

    warped1 = cv2.warpPerspective(image1, H1, (w * 3, h), flags=cv2.INTER_NEAREST)
    warped2 = cv2.warpPerspective(image2, H2, (w * 3, h), flags=cv2.INTER_NEAREST)
    warped3 = cv2.warpPerspective(image3, H3, (w * 3, h), flags=cv2.INTER_NEAREST)

    merged = warped2
    merged[:, :w] = warped1[:, :w]
    merged[:, w * 2:w * 3] = warped3[:, w * 2:w * 3]
    return merged
