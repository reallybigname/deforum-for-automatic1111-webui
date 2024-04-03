import cv2
import numpy as np
import torch

# IN PROGRESS
# reallybigname - auto-navigation functions  
# usage:
# if auto_rotation:
#    rot_mat = rotate_camera_towards_depth(depth_tensor, auto_rotation_steps, w, h, fov_deg, auto_rotation_depth_target)
def rotate_camera_towards_depth(depth_tensor, turn_weight, width, height, h_fov=60, target_depth=1):
    # Compute the depth at the target depth
    target_depth_index = int(target_depth * depth_tensor.shape[0])
    target_depth_values = depth_tensor[target_depth_index]
    max_depth_index = torch.argmax(target_depth_values).item()
    max_depth_index = (max_depth_index, target_depth_index)
    max_depth = target_depth_values[max_depth_index[0]].item()

    # Compute the normalized x and y coordinates
    x, y = max_depth_index
    x_normalized = (x / (width - 1)) * 2 - 1
    y_normalized = (y / (height - 1)) * 2 - 1

    # Calculate horizontal and vertical field of view (in radians)
    h_fov_rad = np.radians(h_fov)
    aspect_ratio = width / height
    v_fov_rad = h_fov_rad / aspect_ratio

    # Calculate the world coordinates (x, y) at the target depth
    x_world = np.tan(h_fov_rad / 2) * max_depth * x_normalized
    y_world = np.tan(v_fov_rad / 2) * max_depth * y_normalized

    # Compute the target position using the world coordinates and max_depth
    target_position = np.array([x_world, y_world, max_depth])

    # Assuming the camera is initially at the origin, and looking in the negative Z direction
    cam_position = np.array([0, 0, 0])
    current_direction = np.array([0, 0, -1])

    # Compute the direction vector and normalize it
    direction = target_position - cam_position
    # direction /= np.linalg.norm(direction)
    direction /= torch.norm(direction)

    # Compute the rotation angle based on the turn_weight (number of frames)
    # axis = torch.cross(current_direction, direction)
    axis = np.cross(current_direction, direction)
    # axis = axis / np.linalg.norm(axis)
    axis = axis / torch.norm(axis)
    # angle = np.arcsin(np.linalg.norm(axis))
    angle = torch.asin(torch.norm(axis))
    max_angle = np.pi * (0.1 / turn_weight)  # Limit the maximum rotation angle to half of the visible screen
    # rotation_angle = np.clip(np.sign(np.cross(current_direction, direction)) * angle / turn_weight, -max_angle, max_angle)
    rotation_angle = torch.clamp(torch.sign(torch.cross(current_direction, direction)) * angle / turn_weight, -max_angle, max_angle)

    # Compute the rotation matrix
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ]) + (1 - np.cos(rotation_angle)) * np.outer(axis, axis)

    # Convert the NumPy array to a PyTorch tensor
    rotation_matrix_tensor = torch.from_numpy(rotation_matrix).float()

    # Add an extra dimension to match the expected shape (1, 3, 3)
    rotation_matrix_tensor = rotation_matrix_tensor.unsqueeze(0)

    return rotation_matrix_tensor


# Asked Bing to give me an explicit version of the function at the end
def rotation_matrix(axis, angle):
    # Convert the axis to a numpy array
    axis = np.asarray(axis)
    
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute the four elements of the quaternion representation of the rotation
    quaternion_a = np.cos(angle / 2.0)
    quaternion_b, quaternion_c, quaternion_d = -axis * np.sin(angle / 2.0)
    
    # Compute the squares of the quaternion elements
    quaternion_a_squared = quaternion_a * quaternion_a
    quaternion_b_squared = quaternion_b * quaternion_b
    quaternion_c_squared = quaternion_c * quaternion_c
    quaternion_d_squared = quaternion_d * quaternion_d
    
    # Compute products of pairs of the quaternion elements
    bc_product, ad_product, ac_product, ab_product, bd_product, cd_product = \
        quaternion_b * quaternion_c, \
        quaternion_a * quaternion_d, \
        quaternion_a * quaternion_c, \
        quaternion_a * quaternion_b, \
        quaternion_b * quaternion_d, \
        quaternion_c * quaternion_d
    
    # Return the rotation matrix derived from the computed elements
    return np.array([
        [quaternion_a_squared + quaternion_b_squared - quaternion_c_squared - quaternion_d_squared,
         2 * (bc_product + ad_product),
         2 * (bd_product - ac_product)],
        
        [2 * (bc_product - ad_product),
         quaternion_a_squared + quaternion_c_squared - quaternion_b_squared - quaternion_d_squared,
         2 * (cd_product + ab_product)],
        
        [2 * (bd_product + ac_product),
         2 * (cd_product - ab_product),
         quaternion_a_squared + quaternion_d_squared - quaternion_b_squared - quaternion_c_squared]
    ])

    # difficult-to-understand function
    #--------------------------------------
    # def rotation_matrix(axis, angle):
    #     axis = np.asarray(axis)
    #     axis = axis / np.linalg.norm(axis)
    #     a = np.cos(angle / 2.0)
    #     b, c, d = -axis * np.sin(angle / 2.0)
    #     aa, bb, cc, dd = a * a, b * b, c * c, d * d
    #     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    #     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    #                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    #                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def decompose_perspective_matrix(matrix, dimensions):
    # The perspective transformation matrix needs to be 3x3
    matrix_3x3 = np.zeros((3, 3))
    matrix_3x3[:3, :3] = matrix[:, :3]

    # Decompose the perspective transformation matrix
    _, rotation_matrix, translation_vector, _, _, _, _ = cv2.decomposeProjectionMatrix(matrix_3x3)

    # Convert rotation matrix to Euler angles
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    rotation_x, rotation_y, rotation_z = map(np.rad2deg, rotation_vector.flatten())

    # Calculate FOV
    fx = matrix[0, 0]
    fy = matrix[1, 1]
    width, height = dimensions
    fov_x = 2 * np.rad2deg(np.arctan(0.5 * width / fx))
    fov_y = 2 * np.rad2deg(np.arctan(0.5 * height / fy))

    # Return a dictionary with all the keys
    return {
        'translation_x': translation_vector[0],
        'translation_y': translation_vector[1],
        'translation_z': translation_vector[2],
        'rotation_x': rotation_x,
        'rotation_y': rotation_y,
        'rotation_z': rotation_z,
        'fov_x': fov_x,
        'fov_y': fov_y,
    }

def decompose_affine_matrix(matrix):
    # Decompose the affine transformation matrix
    sx = np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
    sy = np.sqrt(matrix[0, 1]**2 + matrix[1, 1]**2)

    # Calculate zoom (scale factor)
    zoom = sx if sx > sy else sy

    # Calculate rotation angle in degrees
    angle = np.rad2deg(np.arctan2(matrix[1, 0] / sx, matrix[0, 0] / sx))

    # Get translation
    translation_x = matrix[0, 2]
    translation_y = matrix[1, 2]

    # Return a dictionary with all the keys
    return {
        'zoom': zoom,
        'angle': angle,
        'translation_x': translation_x,
        'translation_y': translation_y,
    }