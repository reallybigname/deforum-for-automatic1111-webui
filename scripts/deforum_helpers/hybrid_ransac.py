import copy
import cv2
import numpy as np
from .image_functions import bgr2gray
from .hybrid_video import get_resized_image_from_filename
from .general_utils import debug_print

def get_matrix_for_hybrid_motion(idx1, idx2, dimensions, inputfiles, motion, use_prev_img=False, prev_img=None, suppress_console=False):
    frame_msg = f"previous image frame {idx1} to video frame {idx2}" if use_prev_img else f"video frames {idx1} to {idx2}"
    debug_print(f"Calculating {motion} RANSAC matrix from {frame_msg} ", not suppress_console)

    # first handle invalid images by returning default matrix
    # if (use_prev_img and prev_img is None) or (prev_img.dtype != np.uint8 or prev_img.size == 0):
    #     M = get_default_matrix(motion)
    # else:
    if use_prev_img:
        img1 = prev_img.astype(np.uint8)
    else:
        img1 = get_resized_image_from_filename(str(inputfiles[idx1]), dimensions)
    img2 = get_resized_image_from_filename(str(inputfiles[idx2]), dimensions)
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    M = get_transformation_matrix_from_images(img1, img2, motion)
    return M

def image_transform_ransac(image_cv2, M, motion):
    if motion == "Perspective":
        return image_transform_perspective(image_cv2, M)
    else: # Affine
        return image_transform_affine(image_cv2, M)

def image_transform_perspective(image_cv2, M):
    return cv2.warpPerspective(
        image_cv2,
        M,
        (image_cv2.shape[1], image_cv2.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
        flags=cv2.INTER_LANCZOS4
    )

def image_transform_affine(image_cv2, M):
    return cv2.warpAffine(
        image_cv2,
        M,
        (image_cv2.shape[1],image_cv2.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
        flags=cv2.INTER_LANCZOS4
    )

def get_default_matrix(motion):
    return np.eye(3 if motion == "Perspective" else 2, 3)

def get_transformation_matrix_from_images(img1, img2, motion, feature_detector_type='SIFT', max_features=500, max_matches=50, confidence=0.75, ransacReprojThreshold=2.0):
    # Choose the feature detector: SIFT, ORB, AKAZE, or BRISK
    if feature_detector_type == 'SIFT': # slowest, most accurate
        feature_detector = cv2.SIFT_create(nfeatures=max_features)
    elif feature_detector_type == 'AKAZE': # faster than SIFT
        feature_detector = cv2.AKAZE_create()
    elif feature_detector_type == 'ORB': # lower quality, fast, better for realtime applications
        feature_detector = cv2.ORB_create(nfeatures=max_features)
    elif feature_detector_type == 'BRISK': # lower quality, fast, better for realtime applications
        feature_detector = cv2.BRISK_create(nfeatures=max_features)
    else:
        raise ValueError("Invalid feature detector type. Choose from 'SIFT', 'ORB', 'AKAZE', or 'BRISK'.")

    try:
        # Detect and compute keypoints and descriptors
        kp1, des1 = feature_detector.detectAndCompute(img1, None)
        kp2, des2 = feature_detector.detectAndCompute(img2, None)

        # Match features using KNN or BFMatcher based on the feature detector
        if feature_detector_type in ['SIFT', 'AKAZE']:
            # Create BFMatcher object and match descriptors
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test to filter good matches
            good_matches = []
            for match in matches:
                # Check if the match has 2 elements
                if len(match) == 2:
                    m, n = match
                    if m.distance < confidence * n.distance:
                        good_matches.append(m)
                # Optionally, handle the case where len(match) != 2
                # e.g., log a warning or take some other action
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

        if len(good_matches) <= 8:
            return get_default_matrix(motion)

        # Convert keypoints to numpy arrays
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if len(src_pts) <= 8 or len(dst_pts) <= 8:
            return get_default_matrix(motion)
        elif motion == "Perspective":  # Perspective transformation (3x3)
            transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
            return transformation_matrix if is_valid_transformation_matrix(transformation_matrix, motion) else get_default_matrix(motion)
        else:  # Affine - rigid transformation (no skew 3x2)
            transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            return transformation_rigid_matrix if is_valid_transformation_matrix(transformation_rigid_matrix, motion) else get_default_matrix(motion)
    except Exception as e:
        print(f"Using default matrix after an error getting {motion} motion matrix: {e}")
        # Return a default matrix in case of failure
        return get_default_matrix(motion)

def is_valid_transformation_matrix(matrix, motion):
    if matrix is None:
        return False
    if isinstance(matrix, np.ndarray):
        ok = matrix.dtype in [np.float32, np.float64]
    elif isinstance(matrix, cv2.UMat):
        ok = matrix.type() in [cv2.CV_32F, cv2.CV_64F]
    if matrix.shape == (2 if motion == "Affine" else 3, 3):
        return True if ok else False
    
def blend_matrices(matrix1, alpha, matrix2, beta):
    return (alpha * matrix1) + (beta * matrix2)

def calculate_transformation_magnitude(matrix):
    # Create an identity matrix of the same size as your input matrix
    identity = np.eye(matrix.shape[0], matrix.shape[1])
    
    # Calculate the difference between your matrix and the identity matrix
    diff = matrix - identity
    
    # Calculate the Frobenius norm of the difference
    magnitude = np.linalg.norm(diff, 'fro')
    
    # Return the magnitude
    return magnitude

def combine_matrices(weights, matrices):
    if len(matrices) == 0:
        raise ValueError("At least one matrix must be provided.")
    if len(matrices) != len(weights):
        raise ValueError("The number of weights must match the number of matrices.")
    dimensions = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != dimensions:
            raise ValueError("All matrices must have the same dimensions.")
    combined_matrix = np.zeros(dimensions, dtype=np.float32)
    for weight, matrix in zip(weights, matrices):
        if weight < 0:
            weight = 0
        combined_matrix += weight * matrix
    combined_matrix /= sum(weights)
    return combined_matrix

# does affine or perspective
def invert_matrix(matrix):
    # Check the shape of the matrix
    if matrix.shape == (2, 3):
        # Convert 2x3 matrix to 3x3
        matrix = np.vstack((matrix, [0, 0, 1]))
    elif matrix.shape != (3, 3):
        raise ValueError("Input matrix must be either 2x3 or 3x3")

    # Compute the inverse
    try:
        inverse_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("The input matrix is singular and does not have an inverse")

    # If the original matrix was 2x3, convert the inverse back to 2x3
    if matrix.shape == (3, 3) and all(matrix[2, :] == np.array([0, 0, 1])):
        inverse_matrix = inverse_matrix[:2, :]

    return inverse_matrix