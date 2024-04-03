import io
import cv2
import cv2.cuda
import numpy as np
import copy
import torch
from .image_functions import bgr2gray
from .consistency_check import make_consistency
from .hybrid_video import get_resized_image_from_filename
from .hybrid_ransac import invert_matrix
from .hybrid_flow_vis import save_flow_visualization
from .updown_scale import updown_scale_whatever
from .general_utils import debug_print

def get_flow_for_hybrid_motion(idx1, idx2, dimensions, inputfiles, hybridframes_path, prev_flow, hybrid_motion, method, mode, raft_model, depth_model, depth, use_prev_img=False, prev_img=None,
                               consistency_check=False, consistency_blur=0, do_flow_visualization=False, updown_scale=1, matrix_flow=None, anchoring=False, img=None, suppress_console=False):
    # switcher for warp function which takes same params
    warp_function = cv2.warpAffine if mode == '2D' else cv2.warpPerspective
    
    frame_msg = f"previous image frame {idx1} to video frame {idx2}" if use_prev_img else f"video frames {idx1} to {idx2}"
    debug_print(f"Calculating {method} optical flow{' w/consistency mask' if consistency_check else ''} from {frame_msg}", suppress_console)

    if matrix_flow is not None:
        mf_matrix = matrix_flow[0]

    if updown_scale > 1:
        # upscale dimensions and prev_img if needed]
        shape_img = (dimensions[1], dimensions[0])  # (h, w)
        # change dimensions for pulling images from files (instead of relying on the flow function to resize the images)
        original_dimensions = copy.deepcopy(dimensions)
        dimensions = (int(dimensions[0]*updown_scale), int(dimensions[1]*updown_scale)) # (w, h)
        # upscale prev_img because it's not from a file (files handles below)
        if use_prev_img:
            prev_img, shape_img = updown_scale_whatever(prev_img, scale=updown_scale)
        if matrix_flow is not None:
            mf_matrix, _ = updown_scale_whatever(mf_matrix, scale=updown_scale)
        if img is not None:
            img, _ = updown_scale_whatever(img, scale=updown_scale)

    if use_prev_img and prev_img is None:
        # return default still motion if no prev_img is available
        flow = get_default_flow(dimensions)
    else:
        # get video frames - if use_prev_img we use the previous image instead of the previous video frame 
        last_image = prev_img.astype(np.uint8) if use_prev_img else get_resized_image_from_filename(str(inputfiles[idx1]), dimensions)
        this_img = get_resized_image_from_filename(str(inputfiles[idx2]), dimensions)

        # get the optical flow
        flow_args = (last_image, this_img, method, raft_model, prev_flow, consistency_check, consistency_blur)
        flow = get_flow_from_images(*flow_args, anchoring=anchoring)

        # IN PROGRESS
        # adjust flow to include the displacements of rendered shapes vs video shapes
        if img is not None:
            flow = adjust_flow_for_altered_frame(flow, last_image, img)
            flow = filter_chaotic_flow(flow, magnitude_threshold=10, consistency_threshold=4)

        # counter movement of flow in viewport by warping the flow with inverse ransac matrix
        if hybrid_motion == 'Matrix Flow' and matrix_flow is not None:
            warpkwargs = {'borderMode': cv2.BORDER_REFLECT_101, 'flags': cv2.INTER_LANCZOS4}
            flow = warp_function(flow, invert_matrix(mf_matrix), dimensions, **warpkwargs)

    # downscale flow
    if updown_scale > 1:
        flow, _ = updown_scale_whatever(flow, shape=shape_img)

    # do flow visualization at original size
    if do_flow_visualization:
        if updown_scale > 1:
            dimensions = original_dimensions
        save_flow_visualization(idx1, dimensions, flow, inputfiles, hybridframes_path, suppress_console=True)

    return flow

def image_transform_optical_flow(img, flow, flow_factor=1.0):
    # if flow factor not normal, calculate flow factor
    if flow_factor != 1:
        flow *= flow_factor

    # the flow is reversed when calling the flow routines normally
    # it annoyed me to send the images backwards to the function, so I reverse it here instead
    flow = -flow

    h, w = img.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]
    return remap(img, flow)

def get_default_flow(dimensions):
    cols, rows = dimensions
    flow = np.zeros((rows, cols, 2), np.float32)
    return flow

def get_flow_from_images(img1, img2, method, raft_model, prev_flow=None, consistency=False, consistency_blur=0, reliability=0, anchoring=False, low_flow_priority=False, low_threshold=0, high_threshold=1, updown_scale=1):
    # upscale images
    if updown_scale > 1:
        img1, _ = updown_scale_whatever(img1, scale=updown_scale)
        img2, shape = updown_scale_whatever(img2, scale=updown_scale)
   
    # reliable flow init, args/kwargs init
    reliable_flow = None
    get_any_flow_args = (img1, img2, method, raft_model)
    get_any_flow_args_reverse = (img2, img1, method, raft_model)
    get_any_flow_kwargs = {'anchoring': anchoring, 'low_flow_priority': low_flow_priority, 'low_threshold': low_threshold, 'high_threshold': high_threshold}

    # normal or consistency flows require flow forward
    flow_forward = get_any_flow_from_images(*get_any_flow_args, prev_flow, **get_any_flow_kwargs)

    # if not using consistency mask (reliable flow)
    if not consistency:
        flow = flow_forward
    else:
        # consistency check needs backwards flow and gets reliable_flow using forward and backward
        flow_backward = get_any_flow_from_images(*get_any_flow_args_reverse, None, **get_any_flow_kwargs)

        # get reliable flow consistency mask
        reliable_flow = make_consistency(flow_forward, flow_backward, edges_unreliable=False)

        # optional blur of consistency mask
        if consistency_blur > 0:
            reliable_flow = box_blur(reliable_flow.astype(np.float32), consistency_blur)

        # final filtered flow using reliable flow
        flow = filter_flow(flow_forward, reliable_flow, consistency_blur, reliability)

    # downscale flows
    if updown_scale > 1:
        flow, _ = updown_scale_whatever(flow, shape=shape)
        if reliable_flow is not None:
            reliable_flow, _ = updown_scale_whatever(flow, shape=shape)

    return flow

def box_blur(input_array, radius):
    # Convert the input array to a float32 array
    input_array = np.float32(input_array)
    # Apply the box filter
    result = cv2.boxFilter(input_array, -1, (radius, radius), normalize=True)    
    return result

def filter_flow(flow, reliable_flow, reliability=0.5, consistency_blur=0):
    # reliability from reliabile flow: -0.75 is bad, 0 is meh/outside, 1 is great
    # Create a mask from the first channel of the reliable_flow array
    mask = reliable_flow[..., 0]

    # to set everything to 1 or 0 based on reliability (not as good)
    # mask = np.where(mask >= reliability, 1, 0)

    # Expand the mask to match the shape of the forward_flow array
    mask = np.repeat(mask[..., np.newaxis], flow.shape[2], axis=2)

    # Apply the mask to the flow
    return flow * mask

def feature_anchored_optical_flow(img1, img2, flow, use_sift=False, confidence=0.05, max_adjustments=5, max_features=1000, max_matches=100, desired_good_matches=50):
    try:
        # Feature detector initialization
        feature_detector = cv2.SIFT_create(nfeatures=max_features) if use_sift else cv2.ORB_create(nfeatures=max_features)
        
        # Detect features
        kp1, des1 = feature_detector.detectAndCompute(img1, None)
        kp2, des2 = feature_detector.detectAndCompute(img2, None)

        # Check if descriptors are empty
        if des1 is None or des2 is None or not des1.any() or not des2.any():
            print("No features detected.")
            return flow

        # Matcher initialization
        bf = cv2.BFMatcher(cv2.NORM_HAMMING if not use_sift else cv2.NORM_L2, crossCheck=not use_sift)
        if not use_sift:
            des1, des2 = np.float32(des1), np.float32(des2)

        for adjustment_iteration in range(max_adjustments):
            # Match features and apply filtering criteria
            matches = bf.knnMatch(des1, des2, k=2) if use_sift else bf.match(des1, des2)
            good_matches = [m for m, n in matches if m.distance < confidence * n.distance] if use_sift else sorted(matches, key=lambda x: x.distance)[:max_matches]
            
            if len(good_matches) >= desired_good_matches:
                break

            # Adjust parameters for next iteration
            max_features += 100
            max_matches += 10

        # Extract matched keypoints
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

        # Adjust the flow using feature matches
        for p1, p2 in zip(pts1, pts2):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            flow[y1, x1] = np.array([x2 - x1, y2 - y1])

        return flow
    except cv2.error as cv_err:
        print(f"OpenCV Error: {cv_err}")
        return flow
    except Exception as e:
        print(f"Unexpected error: {e}")
        return flow

def get_flow_from_images_RAFT(img1, img2, raft_model):
    flow = raft_model.predict(img1, img2)
    return flow

def get_flow_from_images_DIS(img1, img2, preset, prev_flow):
    def set_dis(d, grad_desc=25, finest_scale=1, patch_size=8, patch_stride=4, variational_refinement_iterations=None,
                variational_refinement_alpha=None, variational_refinement_delta=None, variational_refinement_gamma=None):
        d.setGradientDescentIterations(grad_desc)
        d.setFinestScale(finest_scale)
        d.setPatchSize(patch_size)
        d.setPatchStride(patch_stride)
        if variational_refinement_iterations is not None:
            d.setVariationalRefinementIterations(variational_refinement_iterations) # 10
        if variational_refinement_alpha is not None:
            d.setVariationalRefinementAlpha(variational_refinement_alpha) # 0.01
        if variational_refinement_delta is not None:
            d.setVariationalRefinementDelta(variational_refinement_delta) # 0
        if variational_refinement_gamma is not None:
            d.setVariationalRefinementGamma(variational_refinement_gamma) # 0

    # DIS presets chart key: finest scale, grad desc its, patch size, patch stride
    # DIS_MEDIUM: 25, 1, 8, 4 | DIS_FAST: 16, 2, 8, 4 | DIS_ULTRAFAST: 12, 2, 8, 4
    if preset == 'medium': preset_code = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    elif preset == 'fast': preset_code = cv2.DISOPTICAL_FLOW_PRESET_FAST
    elif preset == 'ultrafast': preset_code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    else: preset_code = None
    dis = cv2.DISOpticalFlow_create(preset_code)

    # custom presets
    if preset == 'slow':
        set_dis(dis, 35, 1, 8, 4)
    elif preset == 'fine':
        set_dis(dis, 35, 0, 8, 2)
    elif preset == 'ultrafine':
        set_dis(dis, 75, 0, 11, 1, variational_refinement_iterations=20, variational_refinement_alpha=0.001, variational_refinement_delta=0.001, variational_refinement_gamma=0.01)

    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)

    return dis.calc(img1, img2, prev_flow)

def get_flow_from_images_Farneback(img1, img2, last_flow=None, preset="normal", pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0):
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN         # Specify the operation flags
    pyr_scale = 0.5   # The image scale (<1) to build pyramids for each image
    if preset == "fine":
        levels = 13       # The number of pyramid layers, including the initial image
        winsize = 77      # The averaging window size
        iterations = 13   # The number of iterations at each pyramid level
        poly_n = 15       # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 0.8  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    else: # "normal"
        levels = 5        # The number of pyramid layers, including the initial image
        winsize = 21      # The averaging window size
        iterations = 5    # The number of iterations at each pyramid level
        poly_n = 7        # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 1.2  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    flags = 0 # flags = cv2.OPTFLOW_USE_INITIAL_FLOW    
    flow = cv2.calcOpticalFlowFarneback(img1, img2, last_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    return flow

# NVIDIA HW FLOW
def get_flow_from_images_NvidiaOpticalFlowBM(img1, img2, prev_flow=None, block_size=16, shift_size=1, max_range=64, use_initial_flow=False):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    flow = cv2.cuda.OpticalFlowBM_create(block_size=block_size, shift_size=shift_size, max_range=max_range, use_initial_flow=use_initial_flow)
    if prev_flow is not None:
        flow.setFlowSmooth(prev_flow)
    return flow.calc(img1, img2)

def get_flow_from_images_NvidiaOpticalFlowDual_TVL1(img1, img2, prev_flow=None, tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01, iterations=300, use_initial_flow=False):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    flow = cv2.cuda.OpticalFlowDual_TVL1_create(tau=tau, lambda_=lambda_, theta=theta, nscales=nscales, warps=warps, epsilon=epsilon, iterations=iterations, use_initial_flow=use_initial_flow)
    if prev_flow is not None:
        flow.setFlowSmooth(prev_flow)
    return flow.calc(img1, img2)

def get_flow_from_images_NvidiaOpticalFlowPyrLK(img1, img2, prev_flow=None, winSize=(21, 21), maxLevel=3, iters=30, use_initial_flow=False):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    flow = cv2.cuda.OpticalFlowPyrLK_create(winSize=winSize, maxLevel=maxLevel, iters=iters, use_initial_flow=use_initial_flow)
    if prev_flow is not None:
        flow.setFlowSmooth(prev_flow)
    return flow.calc(img1, img2)

def get_flow_from_images_NvidiaOpticalFlowFarneback(img1, img2, prev_flow=None, pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0, use_initial_flow=False):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    flow = cv2.cuda.OpticalFlowFarneback_create(pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags, use_initial_flow=use_initial_flow)
    if prev_flow is not None:
        flow.setFlowSmooth(prev_flow)
    return flow.calc(img1, img2)

def get_flow_from_images_NvidiaOpticalFlowDeepFlow(img1, img2, prev_flow=None, use_initial_flow=False):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    flow = cv2.cuda.OpticalFlowDeepFlow_create(use_initial_flow=use_initial_flow)
    if prev_flow is not None:
        flow.setFlowSmooth(prev_flow)
    return flow.calc(img1, img2)

# BEGIN CONTRIB-ONLY FLOW FUNCTIONS - These only show up and will only work with cv2.optflow, which is only available if contrib version of cv2 is installed
def get_flow_from_images_Dense_RLOF(img1, img2, last_flow=None):
    return cv2.optflow.calcOpticalFlowDenseRLOF(img1, img2, flow = last_flow)
def get_flow_from_images_SF(img1, img2, last_flow=None, layers = 3, averaging_block_size = 2, max_flow = 4):
    return cv2.optflow.calcOpticalFlowSF(img1, img2, layers, averaging_block_size, max_flow)
def get_flow_from_images_DualTVL1(img1, img2, last_flow=None):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    f = cv2.optflow.DualTVL1OpticalFlow_create()
    return f.calc(img1, img2, last_flow)
def get_flow_from_images_DeepFlow(img1, img2, last_flow=None):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    f = cv2.optflow.createOptFlow_DeepFlow()
    return f.calc(img1, img2, last_flow)
def get_flow_from_images_PCAFlow(img1, img2, last_flow=None):
    img1 = bgr2gray(img1)
    img2 = bgr2gray(img2)
    f = cv2.optflow.createOptFlow_PCAFlow()
    return f.calc(img1, img2, last_flow)
# END CONTRIB-ONLY FLOW FUNCTIONS

def get_flow_methods(flow_group=None):
    all = {}
    group_main = {}
    group_optflow = {}
    group_nvidia = {}

    # available with opencv or opencv-contrib (non-contrib or contrib)
    group_main.update({
        "DIS UltraFast": get_flow_from_images_DIS,
        "DIS Fast": get_flow_from_images_DIS,
        "DIS Medium": get_flow_from_images_DIS,
        "DIS Slow": get_flow_from_images_DIS,
        "DIS Fine": get_flow_from_images_DIS,
        "DIS UltraFine": get_flow_from_images_DIS,
        "Farneback": get_flow_from_images_Farneback,
        "RAFT": get_flow_from_images_RAFT
    })
    all.update(group_main)
    
    # cv2.optflow requires running opencv-contrib-python INSTEAD of opencv-python
    if hasattr(cv2, 'optflow'):
        group_optflow.update({
            "DeepFlow": get_flow_from_images_DeepFlow,
            "DenseRLOF": get_flow_from_images_Dense_RLOF,
            "DualTVL1": get_flow_from_images_DualTVL1,
            "PCAFlow": get_flow_from_images_PCAFlow,
            "SF": get_flow_from_images_SF
        })
    all.update(group_optflow)

    # Check if CUDA module is available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # cuda flows require cuda support and running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        if hasattr(cv2.cuda, 'OpticalFlowBM_create'):          group_nvidia.update({"NvidiaOpticalFlowBM":        get_flow_from_images_NvidiaOpticalFlowBM        })
        if hasattr(cv2.cuda, 'OpticalFlowDual_TVL1_create'):   group_nvidia.update({"NvidiaOpticalFlowDual_TVL1": get_flow_from_images_NvidiaOpticalFlowDual_TVL1 })
        if hasattr(cv2.cuda, 'OpticalFlowPyrLK_create'):       group_nvidia.update({"NvidiaOpticalFlowPyrLK":     get_flow_from_images_NvidiaOpticalFlowPyrLK     })
        if hasattr(cv2.cuda, 'OpticalFlowFarneback_create'):   group_nvidia.update({"NvidiaOpticalFlowFarneback": get_flow_from_images_NvidiaOpticalFlowFarneback })
        if hasattr(cv2.cuda, 'OpticalFlowDeepFlow_create'):    group_nvidia.update({"NvidiaOpticalFlowDeepFlow":  get_flow_from_images_NvidiaOpticalFlowDeepFlow  })
    all.update(group_nvidia)

    if flow_group == 'main':
        return group_main
    elif flow_group == 'optflow':
        return group_optflow
    elif flow_group == 'nvidia':
        return group_nvidia
    else:
        return all

# so it only loads them once
flow_methods = get_flow_methods()
    
def get_any_flow_from_images(img1, img2, method, raft_model, prev_flow=None, anchoring=False, low_flow_priority=False, low_threshold=0, high_threshold=1):
    # put img args into a var
    imgs = (img1, img2)

    if method in flow_methods:
        # get the function from the dictionary
        flow_func = flow_methods[method]
        # call the function with the appropriate arguments
        if method == "RAFT":
            if raft_model is None:
                raise Exception("RAFT Model not provided to get_flow_from_images function, cannot continue.")
            flow = flow_func(*imgs, raft_model)
        elif method.startswith("DIS "):
            # pass the mode as an extra argument for DIS methods
            mode = method.split()[1]
            flow = flow_func(*imgs, mode.lower(), prev_flow)
        else:
            # for other methods, just pass the images and prev_flow
            flow = flow_func(*imgs, prev_flow)
    else:
        # if we reached this point, something went wrong. raise an error:
        raise RuntimeError(f"Invald flow method name: '{method}'")

    # make low magnitude flows win over high magnitude ones
    if low_flow_priority:
        flow = make_low_flow_magnitude_priority(flow)

    # anchoring flow to features gathered with SIFT or ORB
    if anchoring:
        flow = feature_anchored_optical_flow(*imgs, flow)

    # remove low or high threshold
    if low_threshold > 0 or high_threshold < 1:
        flow = proportionally_adjust_flow(flow, low_threshold=low_threshold, high_threshold=high_threshold)

    return flow

def remap(img, flow):
    border_mode = cv2.BORDER_REFLECT_101
    h, w = img.shape[:2]
    displacement = int(h * 0.25), int(w * 0.25)
    larger_img = cv2.copyMakeBorder(img, displacement[0], displacement[0], displacement[1], displacement[1], border_mode)
    lh, lw = larger_img.shape[:2]
    larger_flow = extend_flow(flow, lw, lh)
    remapped_img = cv2.remap(larger_img, larger_flow, None, cv2.INTER_LANCZOS4, border_mode)
    output_img = center_crop_image(remapped_img, w, h)
    return output_img

def remap_flow(flow, remap_flow):
    # Invert the remapping flow
    remap_flow = -remap_flow
    h, w = flow.shape[:2]

    # Add the corresponding x or y coordinate to each point in the remapping flow matrix
    remap_flow[:,:,0] += np.arange(w)
    remap_flow[:,:,1] += np.arange(h)[:,np.newaxis]

    # Use cv2.remap() to predict the new flow from the current flow and the inverted remapping flow
    # new_flow = cv2.remap(flow, remap_flow, None, cv2.INTER_LANCZOS4)
    new_flow = remap(flow, remap_flow)

    return new_flow

def center_crop_image(img, w, h):
    y, x, _ = img.shape
    width_indent = int((x - w) / 2)
    height_indent = int((y - h) / 2)
    cropped_img = img[height_indent:y-height_indent, width_indent:x-width_indent]
    return cropped_img

def extend_flow(flow, w, h):
    # Get the shape of the original flow image
    flow_h, flow_w = flow.shape[:2]
    # Calculate the position of the image in the new image
    x_offset = int((w - flow_w) / 2)
    y_offset = int((h - flow_h) / 2)
    # Generate the X and Y grids
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    # Create the new flow image and set it to the X and Y grids
    new_flow = np.dstack((x_grid, y_grid)).astype(np.float32)
    # Shift the values of the original flow by the size of the border
    flow[:,:,0] += x_offset
    flow[:,:,1] += y_offset
    # Overwrite the middle of the grid with the original flow
    new_flow[y_offset:y_offset+flow_h, x_offset:x_offset+flow_w, :] = flow
    # Return the extended image
    return new_flow

def normalize_flow(flow, dimensions):
    fx, fy = flow[:,:,0], flow[:,:,1]
    w, h = dimensions
    max_flow_x = np.max(np.abs(fx))
    max_flow_y = np.max(np.abs(fy))
    max_flow = max(max_flow_x, max_flow_y)
    rel_fx = fx / (max_flow * w)
    rel_fy = fy / (max_flow * h)
    return np.dstack((rel_fx, rel_fy))

def denormalize_flow(rel_flow, dimensions):
    rel_fx, rel_fy = rel_flow[:,:,0], rel_flow[:,:,1]
    w, h = dimensions
    max_flow_x = np.max(np.abs(rel_fx * w))
    max_flow_y = np.max(np.abs(rel_fy * h))
    max_flow = max(max_flow_x, max_flow_y)
    fx = rel_fx * (max_flow * w)
    fy = rel_fy * (max_flow * h)
    return np.dstack((fx, fy))

def retain_detail_and_eliminate_chaotic_flow(flow, motion_threshold=10, sigma_spatial=3, sigma_color=0.1):
    """
    Retain fine detail while eliminating chaotic flow in the optical flow field.

    Parameters:
        - flow: Input optical flow field (2-channel array, dtype=float32).
        - motion_threshold: Threshold for identifying regions with significant motion (default: 10).
        - sigma_spatial: Spatial standard deviation for edge-preserving filter (default: 3).
        - sigma_color: Color standard deviation for edge-preserving filter (default: 0.1).

    Returns:
        - filtered_flow: Filtered optical flow field.
    """
    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

    # Apply motion-based filtering
    motion_mask = magnitude > motion_threshold

    # Edge-preserving smoothing using bilateral filter
    smoothed_flow = cv2.ximgproc.guidedFilter(flow, flow, radius=3, eps=0.01)

    # Mask out chaotic flow regions
    filtered_flow = np.where(motion_mask[..., np.newaxis], smoothed_flow, flow)

    return filtered_flow

def denoise_flow_nlmeans(flow, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """
    Apply Non-Local Means Denoising to an optical flow field.

    Parameters:
        - flow: Input optical flow field (2-channel array, dtype=float32).
        - h: Filter strength for luminance component (default: 10).
        - hColor: Filter strength for color components (default: 10).
        - templateWindowSize: Size of the window used to compute weights (default: 7).
        - searchWindowSize: Size of the window used to search for similar patches (default: 21).

    Returns:
        - denoised_flow: Optical flow field after denoising.
    """
    # Convert the flow field to RGB format for denoising
    flow_rgb = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)

    # Convert the flow field to uint8 format
    flow_rgb_uint8 = cv2.normalize(flow_rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply Non-Local Means Denoising
    denoised_flow_rgb_uint8 = cv2.fastNlMeansDenoisingColored(flow_rgb_uint8, None, h, hColor, templateWindowSize, searchWindowSize)

    # Convert the denoised flow field back to float32 format
    denoised_flow = cv2.normalize(denoised_flow_rgb_uint8, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    return denoised_flow

def reduce_flow_noise_gpu(flow, window_size=5):
    """
    Reduce noise in the optical flow field using a median filter on GPU.

    Parameters:
        - flow: Input optical flow field (2-channel array, dtype=float32).
        - window_size: Size of the median filter window. Higher values increase smoothing (default: 5).

    Returns:
        - smoothed_flow: Optical flow field after noise reduction.
    """
    # Split the flow into horizontal and vertical components
    flow_x, flow_y = cv2.cuda.split(flow)

    # Create a CUDA context
    cv2.cuda.setDevice(0)
    stream = cv2.cuda_Stream()

    # Apply median filter to each component separately
    flow_x_smoothed = cv2.cuda.createMedianFilter(cv2.CV_32F, window_size)
    flow_x_smoothed.apply(flow_x, flow_x_smoothed, stream=stream)

    flow_y_smoothed = cv2.cuda.createMedianFilter(cv2.CV_32F, window_size)
    flow_y_smoothed.apply(flow_y, flow_y_smoothed, stream=stream)

    # Merge the smoothed components back into a single flow field
    smoothed_flow = cv2.cuda.merge([flow_x_smoothed, flow_y_smoothed], stream=stream)

    # Download the result from GPU memory to CPU memory
    smoothed_flow = smoothed_flow.download()

    return smoothed_flow

def combine_flow_fields(weights, flow_fields):
    # takes two lists, one for weights and one for flow fields (length needs to match)
    # weights are effectively normalized
    if len(flow_fields) == 0:
        raise ValueError("At least one flow field must be provided.")
    if len(flow_fields) != len(weights):
        raise ValueError("The number of weights must match the number of flow fields.")
    dimensions = flow_fields[0].shape
    for flow_field in flow_fields:
        if flow_field.shape != dimensions:
            raise ValueError("All flow fields must have the same dimensions.")
    combined_flow = np.zeros(dimensions, dtype=np.float32)
    for weight, flow_field in zip(weights, flow_fields):
        if weight < 0:
            weight = 0
        combined_flow += weight * flow_field
    combined_flow /= sum(weights)
    return combined_flow

def blend_flow_fields(flow_field1, alpha, flow_field2, beta):
    if flow_field1.shape != flow_field2.shape:
        raise ValueError("Flow fields must have the same dimensions.")
    if alpha < 0 or alpha > 1 or beta < 0 or beta > 1:
        raise ValueError(f"Alpha/beta values must be between 0 and 1 | Alpha: {alpha} | Beta: {beta}")
    blended_flow = (np.copy(flow_field1) * alpha) + (np.copy(flow_field2) * beta)
    return blended_flow

def stabilize_flow_motion(flow, factor=1):
    return_flow = np.copy(flow)

    # if factor is not 0, then stabilize return_flow
    if factor != 0:
        # Calculate average flow vector
        avg_flow = np.mean(flow, axis=(0, 1))

        # apply stabilizer_factor
        avg_flow *= factor

        # Construct stabilized flow field
        return_flow = flow - avg_flow

    return return_flow

def stabilize_flow_rotation(flow, factor=1):
    return_flow = np.copy(flow)

    # if factor is not 0, then stabilize return_flow
    if factor != 0:
        # Calculate average flow vector
        avg_flow = np.mean(flow, axis=(0, 1))

        # Calculate average rotation angle
        avg_angle = np.arctan2(avg_flow[1], avg_flow[0])

        # Calculate the average rotation-adjusted flow
        avg_magnitude = np.linalg.norm(avg_flow)
        avg_rotated_flow = avg_magnitude * np.array([np.cos(avg_angle), np.sin(avg_angle)])

        # Apply stabilizer_factor
        avg_rotated_flow *= factor

        # Construct stabilized flow field
        return_flow = flow - avg_rotated_flow

    return return_flow

def estimate_z_translation(flow, depth_map):
    # Convert the flow to a PyTorch tensor if it's not already
    if isinstance(flow, np.ndarray):
        flow = torch.from_numpy(flow).to(depth_map.device)

    # Calculate the flow magnitudes
    flow_magnitudes = torch.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Normalize the flow magnitudes and depth map for comparison
    flow_magnitudes = (flow_magnitudes - flow_magnitudes.min()) / (flow_magnitudes.max() - flow_magnitudes.min())
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Calculate the difference between the normalized flow magnitudes and depth map
    diff = flow_magnitudes - depth_map

    # Calculate the average 'z' translation
    # Positive values indicate forward movement (objects moving closer to the camera)
    # Negative values indicate backward movement (objects moving away from the camera)
    z_translation = diff.mean()

    return z_translation.item()

def combine_estimates(scale_change_flow, scale_change_transform, depth_change, 
                      weight_flow=1.0, weight_transform=1.0, weight_depth=1.0):
    # Calculate the weighted sum of the estimates
    weighted_sum = (scale_change_flow * weight_flow +
                    scale_change_transform * weight_transform +
                    depth_change * weight_depth)

    # Calculate the total weight
    total_weight = weight_flow + weight_transform + weight_depth

    # Compute the weighted average
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    
    combined_estimate = weighted_sum / total_weight
    return combined_estimate

def smooth_estimate(current_estimate, prev_estimates):
    # print(f'current_estimate: {current_estimate} | prev_estimates: {prev_estimates}')
    
    # Add the current estimate to the history
    prev_estimates.append(current_estimate)

    # Calculate the moving average
    smoothed_estimate = sum(prev_estimates) / len(prev_estimates)
    return smoothed_estimate

def adjust_flow_for_altered_frame(original_flow, original_frame, altered_frame):
    """
    Adjust the optical flow field based on alterations made to a frame.

    :param original_flow: Optical flow field from the original frame to the next frame.
    :param original_frame: The original previous frame.
    :param altered_frame: The altered version of the previous frame.
    :return: Adjusted optical flow field.
    """
    # Detect significant changes between the original and altered frames
    difference = cv2.absdiff(original_frame, altered_frame)
    threshold = 30  # Threshold for alteration detection
    altered_regions = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY) > threshold

    # Calculate displacement (placeholder approach)
    displacement = cv2.cvtColor(altered_frame, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    displacement = displacement[..., np.newaxis] * np.array([[1, 1]])

    # Adjust the flow
    adjusted_flow = original_flow.copy()
    adjusted_flow[altered_regions] += displacement[altered_regions]

    return adjusted_flow

def filter_chaotic_flow(adjusted_flow, magnitude_threshold=10, consistency_threshold=2):
    """
    Filter out chaotic areas in the optical flow field based on magnitude and consistency.

    :param adjusted_flow: The adjusted optical flow field.
    :param magnitude_threshold: Threshold for filtering based on flow magnitude.
    :param consistency_threshold: Threshold for filtering based on flow consistency.
    :return: Filtered optical flow field.
    """
    # Flow magnitude thresholding
    flow_magnitude = np.sqrt(adjusted_flow[..., 0]**2 + adjusted_flow[..., 1]**2)
    reliable_flow = flow_magnitude < magnitude_threshold

    # Flow consistency check (placeholder approach)
    flow_consistency = cv2.blur(flow_magnitude, (5, 5))  # Using a simple blur to estimate local consistency
    consistent_flow = flow_consistency < consistency_threshold

    # Combine the two criteria
    reliable_and_consistent_flow = np.logical_and(reliable_flow, consistent_flow)

    # Apply the combined mask to the flow
    adjusted_flow[~reliable_and_consistent_flow] = 0

    return adjusted_flow

def estimate_scale_change_from_flow(flow):
    # Assuming flow is a numpy array of shape (H, W, 2)
    # Each entry flow[y, x] is a vector showing motion from (x, y) to (x + flow[y, x, 0], y + flow[y, x, 1])

    # Calculate the divergence of the flow field
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    d_flow_x = cv2.Sobel(flow_x, cv2.CV_64F, 1, 0, ksize=5)
    d_flow_y = cv2.Sobel(flow_y, cv2.CV_64F, 0, 1, ksize=5)
    divergence = d_flow_x + d_flow_y

    # A positive divergence indicates expansion (movement away from the camera)
    # A negative divergence indicates contraction (movement towards the camera)
    scale_change_estimate = np.mean(divergence)
    return scale_change_estimate

def estimate_z_translation_advanced(flow, depth_tensor, prev_estimates=None, transform_matrix=None):
    # Universal depth estimation approach
    scale_change_flow = estimate_scale_change_from_flow(flow)
    transform_analysis = estimate_scale_change_from_transform(transform_matrix)

    # Combine and smooth estimates
    combined_estimate = combine_estimates(scale_change_flow, transform_analysis, depth_tensor)

    if prev_estimates is None:
        final_estimate = combined_estimate
    else:
        final_estimate = smooth_estimate(combined_estimate, prev_estimates)

    return final_estimate

def estimate_scale_change_from_transform(transform_matrix):
    # Assuming transform_matrix is a 2x3 affine or 3x3 perspective transformation matrix
    # Extract scaling factors (this is a simplistic approach)
    scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
    scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
    scale_change_estimate = (scale_x + scale_y) / 2.0 - 1  # Subtract 1 to center around no change
    return scale_change_estimate

def make_low_flow_magnitude_priority(flow, num_masks=64):
    # Calculate the magnitude of the flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Get the height and width of the flow field
    h, w = flow.shape[:2]

    # Create a series of masks based on the flow intensity
    masks = np.linspace(np.min(magnitude), np.max(magnitude), num_masks)

    # Initialize the processed flow with the original flow
    flow_processed = flow.copy()

    # Precompute the end points for each vector in the flow field
    end_points_y = np.clip(np.arange(h).reshape(-1, 1) + flow[..., 1].astype(int), 0, h - 1)
    end_points_x = np.clip(np.arange(w) + flow[..., 0].astype(int), 0, w - 1)

    # Iterate over each mask
    for mask in masks:
        # Create a binary mask where the flow magnitude is greater than the current mask
        binary_mask = magnitude > mask

        # Use vectorized operations to update the flow_processed
        flow_processed[binary_mask] = flow[end_points_y[binary_mask], end_points_x[binary_mask]]

    return flow_processed

def proportionally_adjust_multiple_flows(*flows, low_threshold, high_threshold):
    """
    Adjust multiple optical flows' magnitudes proportionally based on thresholds.

    :param flows: A variable number of optical flow arrays.
    :param low_threshold: Lower threshold for flow magnitude (0 to 1).
    :param high_threshold: Upper threshold for flow magnitude (0 to 1).
    :return: A list of adjusted optical flows.
    """
    adjusted_flows = []

    for flow in flows:
        flow = proportionally_adjust_flow(flow, low_threshold, high_threshold)
        adjusted_flows.append(adjusted_flow)

    return adjusted_flows

def proportionally_adjust_flow(flow, low_threshold, high_threshold):
    """
    Adjustoptical flow magnitudes proportionally based on thresholds.

    :param flow: An optical flow arrays.
    :param low_threshold: Lower threshold for flow magnitude (0 to 1).
    :param high_threshold: Upper threshold for flow magnitude (0 to 1).
    :return: adjusted optical flow.
    """
    # Calculate magnitude and angle of flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize magnitude to [0, 1]
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Scale magnitudes below low_threshold
    scale_low = magnitude < low_threshold
    magnitude[scale_low] = magnitude[scale_low] / low_threshold

    # Scale magnitudes above high_threshold
    scale_high = magnitude > high_threshold
    magnitude[scale_high] = 1 - (1 - magnitude[scale_high]) / (1 - high_threshold)

    # Convert back to Cartesian coordinates
    adjusted_flow = np.zeros_like(flow)
    adjusted_flow[..., 0], adjusted_flow[..., 1] = cv2.polarToCart(magnitude, angle)

    return adjusted_flow

