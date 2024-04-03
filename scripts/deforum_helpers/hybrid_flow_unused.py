''' CRAZY USEFUL BUT UNUSED & UNTESTED FUNCTIONS '''

def calculate_flow_magnitude(flow):
    # Calculate the magnitude of the flow vectors
    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    
    # Return the mean magnitude across the entire image
    return np.mean(magnitude)

def estimate_depth_change(depth_list):
    # Assuming depth_list is a list of depth tensors

    # Validate that we have at least two tensors to compare
    if len(depth_list) < 2:
        raise ValueError("Need at least two depth tensors to estimate depth change.")

    # Calculate depth change for each consecutive pair and take the mean
    depth_changes = []
    for i in range(len(depth_list) - 1):
        depth_change = torch.abs(depth_list[i + 1] - depth_list[i]).mean()
        depth_changes.append(depth_change)

    # Average the depth changes
    average_depth_change = torch.mean(torch.stack(depth_changes)).item()

    return average_depth_change

def warp_flow_with_matrix(flow, H, invert=False):
    # Check if the input matrix is affine and convert to homography if necessary
    if H.shape == (2, 3):
        H = np.vstack((H, [0, 0, 1]))

    # Invert the homography matrix
    H_new = np.linalg.inv(H) if invert else H
    
    # Create grid of points
    h, w = flow.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    grid = np.stack((x, y, np.ones_like(x)), axis=-1)

    # Apply inverse homography
    warped_grid = np.matmul(H_new, grid.reshape(-1, 3).T)
    warped_grid /= warped_grid[2, :]
    warped_grid = warped_grid.T.reshape(h, w, 3)

    # Calculate warped flow
    warped_flow = flow + warped_grid[..., :2] - grid[..., :2]

    return warped_flow

def flow_to_points(flow):
    # Assuming flow is a 2D numpy array representing optical flow
    h, w = flow.shape[:2]
    
    # Create a grid of (x, y) coordinates
    y, x = np.indices((h, w))
    
    # The source points are just the original coordinates
    points_src = np.stack([x.ravel(), y.ravel()], axis=-1)
    
    # The destination points are the original coordinates plus the flow vectors
    points_dst = points_src + flow.reshape(-1, 2)
    
    return points_src, points_dst

def points_to_flow(points_src, points_dst):
    # Compute the flow by subtracting the source points from the destination points
    flow = points_dst - points_src

    # Reshape the flow back to the 2D format with two channels for x and y flow components
    h, w = points_src[-1][0] + 1, points_src[-1][1] + 1  # assuming points_src is sorted
    flow_2d = flow.reshape((h, w, 2))

    return flow_2d

def create_homography_mask_for_flow(flow, H):
    # Check if the input matrix is affine and convert to homography if necessary
    if H.shape == (2, 3):
        H = np.vstack((H, [0, 0, 1]))

    # Create grid of points
    h, w = flow.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    grid = np.stack((x, y, np.ones_like(x)), axis=-1)

    # Apply homography
    warped_grid = np.matmul(H, grid.reshape(-1, 3).T)
    warped_grid /= warped_grid[2, :]
    warped_grid = warped_grid.T.reshape(h, w, 3)

    # Calculate difference between optical flow and homography
    diff = np.linalg.norm(flow - warped_grid[..., :2], axis=-1)

    # Normalize difference to range [0, 255]
    mask = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mask

def apply_flow_mask(flow, mask, low_thresh=0, high_thresh=255, equalize=False, alpha=1.0, beta=0.0):
    # Apply histogram equalization if requested
    if equalize:
        mask = cv2.equalizeHist(mask)

    # Apply contrast adjustment
    mask = cv2.convertScaleAbs(mask, alpha=alpha, beta=beta)

    # Apply thresholding
    _, mask = cv2.threshold(mask, low_thresh, high_thresh, cv2.THRESH_BINARY)

    # Normalize mask to range [0, 1]
    mask = mask / 255.0

    # Create an empty flow field with the same shape as the input flow
    masked_flow = np.zeros_like(flow)

    # Apply the mask to the flow
    for i in range(3):  # assuming flow is a 3-channel image
        masked_flow[..., i] = flow[..., i] * mask

    return masked_flow

def lucas_kanade_optical_flow(img1, img2, window_size):
    """
    Calculate optical flow between two images using the Lucas-Kanade method.
    """
    # Initialize the flow array with zeros
    flow = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)
    # Calculate gradients
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    It = img2 - img1
    
    # Iterate over each pixel in the window
    for i in range(window_size, img1.shape[0] - window_size):
        for j in range(window_size, img1.shape[1] - window_size):
            # Compute local gradients
            Ix_window = Ix[i-window_size:i+window_size+1, j-window_size:j+window_size+1].flatten()
            Iy_window = Iy[i-window_size:i+window_size+1, j-window_size:j+window_size+1].flatten()
            It_window = -It[i-window_size:i+window_size+1, j-window_size:j+window_size+1].flatten()
            
            # Construct matrix A and vector b
            A = np.vstack((Ix_window, Iy_window)).T
            b = It_window
            
            # Solve for the flow vector if A is invertible
            if np.linalg.matrix_rank(A) == 2:
                nu = np.linalg.inv(A.T @ A) @ A.T @ b
                flow[i, j] = [nu[0], nu[1]]
                
    return flow

def horn_schunck_optical_flow(img1, img2, alpha, num_iterations):
    """
    Calculate optical flow between two images using the Horn-Schunck method.
    """
    # Initialize flow vectors
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    
    # Calculate gradients
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)
    It = img2 - img1
    
    # Iterate to refine the flow
    for _ in range(num_iterations):
        # Average u and v
        u_avg = cv2.blur(u, (3, 3))
        v_avg = cv2.blur(v, (3, 3))
        
        # Compute flow update
        flow_update = (Ix * u_avg + Iy * v_avg + It) / (alpha**2 + Ix**2 + Iy**2)
        u = u_avg - Ix * flow_update
        v = v_avg - Iy * flow_update
    
    flow = np.stack((u, v), axis=-1)
    return flow

def depth_aware_flow_modification(flow, depth_map1, depth_map2, depth_threshold):
    """
    Modify the flow vectors to account for depth information.
    """
    # Depth maps are assumed to be single-channel, with the same height and width as the flow
    depth_diff = np.abs(depth_map1 - depth_map2)
    
    # Scale the flow by the inverse of depth difference - this assumes closer objects (small depth values) should have larger flow
    scale_factor = 1 / (depth_diff + depth_threshold)
    
    # Apply the scale factor to the flow
    modified_flow = flow * scale_factor[..., None]
    
    return modified_flow

def calculate_altered_flow(flow_AB, flow_A_altered, flow_B_altered):
    """
    Calculate the flow between altered frames A and B.

    :param flow_AB: The flow from video frame A to B.
    :param flow_A_altered: The flow from video frame A to its altered version.
    :param flow_B_altered: The flow from video frame B to its altered version.
    :return: The flow between altered frames A and B.
    """
    # Invert the flow from A to altered A
    inverted_flow_A_altered = -flow_A_altered

    # Apply this inverse flow to the flow from A to B
    mapped_flow = flow_AB + inverted_flow_A_altered

    # Add this to the flow from B to altered B to get the final flow
    final_flow = mapped_flow + flow_B_altered

    return final_flow

def enhance_rotational_motion_in_flow(optical_flow, curl_threshold=0.1):
    """
    Enhance rotational motion in an optical flow field.

    Args:
    optical_flow (np.ndarray): The input optical flow field.
    curl_threshold (float): Threshold for curl magnitude to identify rotational motion.

    Returns:
    np.ndarray: The enhanced optical flow field.
    """
    # Calculate the curl of the optical flow
    curl = calculate_curl(optical_flow)

    # Create a mask where the curl exceeds the threshold
    rotational_mask = np.abs(curl) > curl_threshold

    # Enhance the flow based on the mask
    enhanced_flow = np.where(rotational_mask[..., None], 2 * optical_flow, optical_flow)

    return enhanced_flow

def calculate_curl(flow):
    """ Calculate the curl of a 2D vector field. """
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    dFx_dy = np.gradient(flow_x, axis=0)
    dFy_dx = np.gradient(flow_y, axis=1)
    curl = dFy_dx - dFx_dy
    return curl

def simulate_particles_flinging(optical_flow, strength=-1):
    """
    Simulate particles being flung out by rotational momentum in an optical flow field.

    Args:
    optical_flow (np.ndarray): The input optical flow field.
    strength (float): Strength factor for the simulation effect.

    Returns:
    np.ndarray: The modified optical flow field.
    """
    # Assuming the rotational center is at the middle of the flow field
    center_y, center_x = optical_flow.shape[0] / 2, optical_flow.shape[1] / 2

    # Create grid of coordinates
    y, x = np.indices((optical_flow.shape[0], optical_flow.shape[1]))
    pos_vectors = np.stack([x - center_x, y - center_y], axis=-1)

    # Normalize position vectors to get radial directions
    radial_directions = pos_vectors / (np.linalg.norm(pos_vectors, axis=-1, keepdims=True) + 1e-5)

    # Increase the flow magnitude in the radial direction
    modified_flow = optical_flow + strength * radial_directions * np.linalg.norm(optical_flow, axis=-1, keepdims=True)

    return modified_flow

def opt_flow_shape(s, t, ratio):
    ''' Optical flow calculation with shape alignment focus '''

    # Convert images to grayscale if they are multi-channel
    if len(s.shape) == 3:
        s_gray = np.mean(s, axis=-1)
    else:
        s_gray = s
    
    if len(t.shape) == 3:
        t_gray = np.mean(t, axis=-1)
    else:
        t_gray = t
    
    # Calculate gradient of the target image
    fy, fx = np.gradient(t_gray)

    # Calculate gradient of the source image
    sy, sx = np.gradient(s_gray)

    # Compute optical flow as the difference between target and source gradients
    u = fx - sx 
    v = fy - sy 
    
    # Scale the optical flow by the ratio to control the morphing speed
    u *= ratio
    v *= ratio
    
    return np.dstack((u, v))

def compute_canny_edges(img1, img2):
    # Ensure both images have the same dimensions
    assert img1.shape[:2] == img2.shape[:2], "Images must have the same dimensions"

    # Convert images to grayscale
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Automatically detect suitable thresholds using Otsu's method
    low_threshold, _ = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_threshold = 2 * low_threshold  # Using a simple multiplier for the high threshold

    # Compute Canny edges for both frames
    edges1 = cv2.Canny(gray_img1, low_threshold, high_threshold)
    edges2 = cv2.Canny(gray_img2, low_threshold, high_threshold)

    # Dilate edges to make them wider
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges1 = cv2.dilate(edges1, kernel, iterations=1)
    dilated_edges2 = cv2.dilate(edges2, kernel, iterations=1)

    return dilated_edges1, dilated_edges2

def modify_flow_based_on_edges(flow, frame_count, frame_number, canny1, canny2):
    # Ensure both Canny edge maps have the same dimensions
    assert canny1.shape == canny2.shape, "Canny edge maps must have the same dimensions"

    # Calculate step size
    step_size = frame_count - 1

    # Calculate the frame index
    frame_index = int(frame_number * (canny1.shape[0] - 1) / step_size)

    # Select the flow corresponding to the current frame index
    selected_flow = flow[frame_index]

    # Resize the selected flow to match the dimensions of the combined Canny edges
    selected_flow_resized = cv2.resize(selected_flow, (canny1.shape[1], canny1.shape[0]))

    # Ensure canny1 and canny2 have the same number of channels as selected_flow_resized
    if selected_flow_resized.ndim == 2:  # Grayscale image
        canny1 = canny1.squeeze(axis=-1)
        canny2 = canny2.squeeze(axis=-1)

    # Combine dilated edges with flow vectors for the specific frame
    modified_flow = selected_flow_resized * ((canny1 + canny2) / 255.0)

    return modified_flow.astype(flow.dtype)

def remove_outliers(current_frame, previous_frame, next_frame, threshold):
    diff_prev = cv2.absdiff(current_frame, previous_frame)
    diff_next = cv2.absdiff(current_frame, next_frame)
    mask = np.logical_and(diff_prev > threshold, diff_next > threshold)
    return np.where(mask, (previous_frame + next_frame) // 2, current_frame)

def patch_based_artifact_correction_optimized(current_frame, previous_frame, next_frame, artifact_mask, patch_size=5):
    corrected_frame = current_frame.copy()

    # Compute integral images for fast area sum calculation
    int_diff_prev = cv2.integral(cv2.absdiff(current_frame, previous_frame))
    int_diff_next = cv2.integral(cv2.absdiff(current_frame, next_frame))

    # Find coordinates of artifacts
    artifact_coords = np.column_stack(np.where(artifact_mask))

    # Process each artifact region
    for y, x in artifact_coords:
        if patch_size <= y < current_frame.shape[0] - patch_size and patch_size <= x < current_frame.shape[1] - patch_size:
            # Define patch region
            y1, y2 = y - patch_size, y + patch_size + 1
            x1, x2 = x - patch_size, x + patch_size + 1

            # Compute sum of absolute differences within the patch using integral images
            sum_diff_prev = int_diff_prev[y2, x2] - int_diff_prev[y1, x2] - int_diff_prev[y2, x1] + int_diff_prev[y1, x1]
            sum_diff_next = int_diff_next[y2, x2] - int_diff_next[y1, x2] - int_diff_next[y2, x1] + int_diff_next[y1, x1]

            # Choose the most similar patch
            if sum_diff_prev < sum_diff_next:
                replacement_patch = previous_frame[y1:y2, x1:x2]
            else:
                replacement_patch = next_frame[y1:y2, x1:x2]

            # Replace artifact patch
            corrected_frame[y1:y2, x1:x2] = replacement_patch

    return corrected_frame





