import cv2
import numpy as np

def create_anaglyph(bgr_image, depth_tensor):
    # Convert PyTorch tensor to numpy array
    depth_map = depth_tensor.cpu().numpy()

    # Normalize depth map to 0-255
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Convert depth map to 3-channel image
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    # Split BGR image and depth map into individual channels
    b_bgr, g_bgr, r_bgr = cv2.split(bgr_image)
    b_depth, g_depth, r_depth = cv2.split(depth_map)

    # Create red channel for anaglyph image (from BGR image)
    r_anaglyph = r_bgr

    # Create green channel for anaglyph image (average of BGR image and depth map)
    g_anaglyph = ((g_bgr.astype('float32') + g_depth.astype('float32')) / 2).astype('uint8')

    # Create blue channel for anaglyph image (from depth map)
    b_anaglyph = b_depth

    # Merge channels to create anaglyph image
    anaglyph_image = cv2.merge([b_anaglyph, g_anaglyph, r_anaglyph])

    return anaglyph_image

def create_stereo_pair(bgr_image, depth_tensor, disparity_scale=1.0):
    # Convert PyTorch tensor to numpy array
    depth_map = depth_tensor.cpu().numpy()

    # Normalize depth map to 0-255
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Convert depth map to disparity map
    disparity_map = 255 - depth_map

    # Scale disparity map
    disparity_map = (disparity_map * disparity_scale).astype(np.uint8)

    # Create stereo pair
    left_image = bgr_image
    right_image = np.roll(bgr_image, disparity_map, axis=1)

    # Concatenate the images side by side
    stereo_image = np.concatenate((left_image, right_image), axis=1)

    return stereo_image

def prepare_for_vr(img1, img2, K1, D1, K2, D2, R, T):
    """
    Prepare a pair of stereo images for VR display.
    
    Parameters:
    img1, img2: The stereo pair of images.
    
    K1, K2: The camera matrices for the two cameras.
    - These are the intrinsic parameters of the two cameras.
    - Each camera matrix K is a 3x3 matrix and is defined as follows:
        K = [[f_x, 0, c_x],
             [0, f_y, c_y],
             [0, 0, 1]]
    - f_x and f_y are the focal lengths of the camera and c_x and c_y are the coordinates of the principal point that is usually at the image center.
    
    D1, D2: The distortion coefficients for the two cameras.
    - These are the lens distortion coefficients for the two cameras.
    - The distortion coefficients are represented as a vector of 4, 5, or 8 parameters: k1, k2, p1, p2, k3 [, k4, k5, k6].
    - Radial distortion causes straight lines to appear curved. Radial distortions are represented by the parameters k1, k2, and k3.
    - Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane. Tangential distortions are represented by the parameters p1 and p2.
    
    R: The rotation matrix between the two cameras.
    - This is a 3x3 matrix that represents the rotation transformation between the two cameras.
    - The rotation matrix R is used to transform coordinates from one camera's 3D space to the other's.
    
    T: The translation vector between the two cameras.
    - This is a 3x1 vector that represents the translation transformation between the two cameras.
    - The translation vector T represents the distance between the two camera centers.
    """
    
    # Stereo rectification
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, D1, K2, D2, img1.shape[:2], R, T)
    
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img1.shape[:2], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img2.shape[:2], cv2.CV_32FC1)
    
    img1_rectified = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    # Combine the rectified images into a single stereo image
    stereo_img = np.concatenate((img1_rectified, img2_rectified), axis=1)
    
    # Apply barrel distortion
    k1, k2, p1, p2, k3 = 0.1, 0.01, 0.001, 0.001, 0.01  # example distortion coefficients
    K = np.eye(3)  # example camera matrix
    stereo_img_distorted = cv2.undistort(stereo_img, K, np.array([k1, k2, p1, p2, k3]))
    
    return stereo_img_distorted