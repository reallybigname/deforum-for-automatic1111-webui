import os
import cv2
import numpy as np
from .image_functions import bgr2gray_bgr
from .general_utils import debug_print

def save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybridframes_path, suppress_console=False):
    flow_img_file = os.path.join(hybridframes_path, f"flow{frame_idx:09}.jpg")
    flow_img = cv2.imread(str(inputfiles[frame_idx]))
    flow_img = cv2.resize(flow_img, dimensions, cv2.INTER_AREA)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2GRAY)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_GRAY2BGR)
    flow_img = draw_flow_lines_in_grid_in_color(flow_img, flow)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(flow_img_file, flow_img)
    debug_print(f"Saved optical flow visualization: {flow_img_file}", not suppress_console)

def draw_flow_lines_in_grid_in_color(img, flow, step=8, magnitude_multiplier=1, min_magnitude = 0, max_magnitude = 10000):
    flow = flow * magnitude_multiplier
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = bgr2gray_bgr(img)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    vis = cv2.add(vis, bgr)

    # Iterate through the lines
    for (x1, y1), (x2, y2) in lines:
        # Calculate the magnitude of the line
        magnitude = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            b = int(bgr[y1, x1, 0])
            g = int(bgr[y1, x1, 1])
            r = int(bgr[y1, x1, 2])
            color = (b, g, r)
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness=1, tipLength=0.1)    
    return vis


# _________________________________________________________
# |           USEFUL BUT UNUSED FUNCTIONS BELOW           |
# ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

# unused function - an alternate way to visualize flow, not in a grid 
def draw_flow_lines_in_color(img, flow, threshold=3, magnitude_multiplier=1, min_magnitude = 0, max_magnitude = 10000):
    # h, w = img.shape[:2]
    vis = img.copy()  # Create a copy of the input image
    
    # Find the locations in the flow field where the magnitude of the flow is greater than the threshold
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    idx = np.where(mag > threshold)

    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV image to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add color from bgr 
    vis = cv2.add(vis, bgr)

    # Draw an arrow at each of these locations to indicate the direction of the flow
    for i, (y, x) in enumerate(zip(idx[0], idx[1])):
        # Calculate the magnitude of the line
        x2 = x + magnitude_multiplier * int(flow[y, x, 0])
        y2 = y + magnitude_multiplier * int(flow[y, x, 1])
        magnitude = np.sqrt((x2 - x)**2 + (y2 - y)**2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            if i % random.randint(100, 200) == 0:
                b = int(bgr[y, x, 0])
                g = int(bgr[y, x, 1])
                r = int(bgr[y, x, 2])
                color = (b, g, r)
                cv2.arrowedLine(vis, (x, y), (x2, y2), color, thickness=1, tipLength=0.25)

    return vis

# unused function, since the flow visualization makes the masking clear
def save_flow_mask_visualization(frame_idx, reliable_flow, hybridframes_path, color=True):
    flow_mask_img_file = os.path.join(hybridframes_path, f"flow_mask{frame_idx:09}.jpg")
    if color:
        # Normalize the reliable_flow array to the range [0, 255]
        normalized_reliable_flow = (reliable_flow - reliable_flow.min()) / (reliable_flow.max() - reliable_flow.min()) * 255
        # Change the data type to np.uint8
        mask_image = normalized_reliable_flow.astype(np.uint8)
    else:
        # Extract the first channel of the reliable_flow array
        first_channel = reliable_flow[..., 0]
        # Normalize the first channel to the range [0, 255]
        normalized_first_channel = (first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255
        # Change the data type to np.uint8
        grayscale_image = normalized_first_channel.astype(np.uint8)
        # Replicate the grayscale channel three times to form a BGR image
        mask_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2)
    cv2.imwrite(flow_mask_img_file, mask_image)
    print(f"Saved mask flow visualization: {flow_mask_img_file}")

# unused function, since the flow visualization shows the reliable flow
def reliable_flow_to_image(reliable_flow):
    # Extract the first channel of the reliable_flow array
    first_channel = reliable_flow[..., 0]
    # Normalize the first channel to the range [0, 255]
    normalized_first_channel = (first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255
    # Change the data type to np.uint8
    grayscale_image = normalized_first_channel.astype(np.uint8)
    # Replicate the grayscale channel three times to form a BGR image
    bgr_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2)
    return bgr_image
