import numpy as np
import cv2
import py3d_tools as p3d # this is actually a file in our /src folder!
from functools import reduce
import math
import torch
from einops import rearrange
from .general_utils import debug_print
from .updown_scale import updown_scale_whatever, updown_scale_to_integer
from .hybrid_render import get_hybrid_matrix_flow
from .hybrid_flow import estimate_z_translation_advanced

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def construct_RotationMatrixHomogenous(rotation_angles):
    assert(type(rotation_angles)==list and len(rotation_angles)==3)
    RH = np.eye(4,4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

# https://en.wikipedia.org/wiki/Rotation_matrix
def getRotationMatrixManual(rotation_angles):
	
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]
    
    phi         = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z
    
    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp
    
    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg
    
    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st
    
    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta]) 
    
    return R

def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    
    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []
    
    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])
    
    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)
    
    return pin, pout


def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    
    # M is to be estimated
    M          = np.eye(4, 4)
    
    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = scale*d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0)
    f          = h+(d/2.0)
    
    # Translation along Z-axis by -h
    T       = np.eye(4,4)
    T[2,3]  = -h
    
    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])
    
    
    # Projection Matrix 
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0
    
    # pythonic matrix multiplication
    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R]) 
    
    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way. 
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
                 [-W/2., H/2., 0.],[ W/2., H/2., 0.],[ W/2.,-H/2., 0.],[-W/2.,-H/2., 0.]
                 ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)
    
    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)
    
    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

    return M33, sideLength

def get_flip_perspective_matrix(W, H, keys, frame_idx):
    perspective_flip_theta = keys.perspective_flip_theta_series[frame_idx]
    perspective_flip_phi = keys.perspective_flip_phi_series[frame_idx]
    perspective_flip_gamma = keys.perspective_flip_gamma_series[frame_idx]
    perspective_flip_fv = keys.perspective_flip_fv_series[frame_idx]
    M,sl = warpMatrix(W, H, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1., perspective_flip_fv);
    post_trans_mat = np.float32([[1, 0, (W-sl)/2], [0, 1, (H-sl)/2]])
    post_trans_mat = np.vstack([post_trans_mat, [0,0,1]])
    bM = np.matmul(M, post_trans_mat)
    return bM

def flip_3d_perspective(anim_args, prev_img_cv2, keys, frame_idx):
    W, H = (prev_img_cv2.shape[1], prev_img_cv2.shape[0])
    border_mode = border_to_cv2_handle(anim_args.border)
    return cv2.warpPerspective(
        prev_img_cv2,
        get_flip_perspective_matrix(W, H, keys, frame_idx),
        (W, H),
        borderMode=border_mode,
        flags=cv2.INTER_LANCZOS4
    )

def anim_frame_warp(img_cv2, anim_args, keys, frame_idx, depth_model=None, depth=None, device='cuda', half_precision=False,
                    inputfiles=None, hybridframes_path=None, prev_flow=None, raft_model=None, prev_img=None, z_translation_list=None):
    # translate updown scale from anim_arg string
    updown_scale = updown_scale_to_integer(anim_args.updown_scale)
    dimensions = (img_cv2.shape[1], img_cv2.shape[0])
    translation_z_estimate = None
    matrix_flow = None

    can_use_depth = (anim_args.use_depth_warping and depth_model is not None)

    # hybrid matrix flow gets matrix and optical flow from video before animation, so it can add to the matrices used by the animation keys
    # also, the mf_flow component plus depth are used for a z_translation extraction from video
    if anim_args.hybrid_motion == 'Matrix Flow':
        matrix_flow = get_hybrid_matrix_flow(frame_idx, anim_args.max_frames, prev_img, dimensions, anim_args.hybrid_motion_use_prev_img, anim_args.hybrid_motion_behavior,
                                             anim_args.hybrid_flow_method, anim_args.animation_mode, inputfiles, hybridframes_path, prev_flow, raft_model, depth_model, depth)
        mf_matrix = matrix_flow[0]
        mf_flow = matrix_flow[1]
    
    # upscaling before other functions are called
    if updown_scale > 1:
        img_cv2, shape = updown_scale_whatever(img_cv2, scale=updown_scale)
        if can_use_depth and depth is not None:
            depth, _ = updown_scale_whatever(depth, scale=updown_scale)
        if matrix_flow is not None:
            mf_matrix, _ = updown_scale_whatever(mf_matrix, scale=updown_scale)
            mf_flow, _ = updown_scale_whatever(mf_flow, scale=updown_scale)
            matrix_flow[0] = mf_matrix
            matrix_flow[1] = mf_flow

    # 2D/3D animation warping
    if anim_args.animation_mode == '2D':
        img = anim_frame_warp_2d(img_cv2, anim_args, keys, frame_idx, updown_scale_translation=updown_scale, matrix_flow=matrix_flow)
    else: # '3D'
        # if the depth model is available
        if can_use_depth:
            # get depth during animation unless it was passed in
            if depth is None:
                depth = depth_model.predict(img_cv2, anim_args.midas_weight, half_precision)

            # matrix flow's flow and depth are used to estimate z motion
            if matrix_flow is not None and depth is not None:
                # translation_z_estimate = estimate_z_translation(mf_flow, depth)
                translation_z_estimate = estimate_z_translation_advanced(matrix_flow[1], depth, prev_estimates=z_translation_list, transform_matrix=matrix_flow[0])

        else:
            depth = None

        # 3d anim converts depth and rewrites the variable here
        img, depth = anim_frame_warp_3d(device, img_cv2, depth, anim_args, keys, frame_idx, updown_scale_translation=updown_scale,
                                                                matrix_flow=matrix_flow, translation_z_estimate=translation_z_estimate)

    # downsampling
    if updown_scale > 1:
        img, _ = updown_scale_whatever(img, shape=shape)
        if depth is not None:
            depth, _ = updown_scale_whatever(depth, shape=shape)

    return img, depth, matrix_flow, translation_z_estimate

def anim_frame_warp_2d(prev_img_cv2, anim_args, keys, frame_idx, updown_scale_translation=1, matrix_flow=None):
    transform_center_x = keys.transform_center_x_series[frame_idx]
    transform_center_y = keys.transform_center_y_series[frame_idx]
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    translation_x = keys.translation_x_series[frame_idx]
    translation_y = keys.translation_y_series[frame_idx]
    
    # images were scaled beforehand, so we need to change translation to match
    if updown_scale_translation > 1:
        translation_x *= updown_scale_translation
        translation_y *= updown_scale_translation
    
    height, width = prev_img_cv2.shape[:2]
    center_point = (width * transform_center_x, height * transform_center_y)
    rot_mat = cv2.getRotationMatrix2D(center_point, angle, zoom)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    if anim_args.enable_perspective_flip:
        bM = get_flip_perspective_matrix(width, height, keys, frame_idx)
        rot_mat = np.matmul(bM, rot_mat, trans_mat)
    else:
        rot_mat = np.matmul(rot_mat, trans_mat)

    # if custom affine matrix is provided, multiply it with the rot_mat
    if matrix_flow is not None:
        rot_mat = np.matmul(np.vstack([matrix_flow[0], [0,0,1]]), rot_mat)
 
    border_mode = border_to_cv2_handle(anim_args.border)
    result = cv2.warpPerspective(
        prev_img_cv2,
        rot_mat,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=border_mode,
        flags=cv2.INTER_LANCZOS4
    )

    return result

def anim_frame_warp_3d(device, prev_img_cv2, depth, anim_args, keys, frame_idx, updown_scale_translation=1, matrix_flow=None, translation_z_estimate=None):
    # original DISCO translation scale - as if images were all 200x200
    # 1 unit of translation is NOT equal to 1px (like it is in 2D mode)
    # with an equal translation in all dimensions, the p3d perspective camera has to be given an aspect ratio of 1 (not width/height!)
    translation_scale = 1.0/200.0 # matches Disco

    # images were scaled beforehand, so we need to change translation to match
    if updown_scale_translation > 1:
        translation_scale *= updown_scale_translation

    translate_xyz = [
        -keys.translation_x_series[frame_idx] * translation_scale,
        keys.translation_y_series[frame_idx] * translation_scale, 
        -keys.translation_z_series[frame_idx] * translation_scale
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]),
        math.radians(keys.rotation_3d_y_series[frame_idx]),
        math.radians(keys.rotation_3d_z_series[frame_idx])
    ]

    if anim_args.enable_perspective_flip:
        prev_img_cv2 = flip_3d_perspective(anim_args, prev_img_cv2, keys, frame_idx)
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)

    # if custom perspective matrix is provided, modify it to the correct shape and then multiply it with the rot_mat
    if matrix_flow is not None:
        rot_mat, translate_xyz = combine_cv2_with_p3d_perspective(matrix_flow[0], translation_scale, rot_mat, translate_xyz, translation_z_estimate, device)

    result, depth_used = transform_image_3d_switcher(device if not device.type.startswith('mps') else torch.device('cpu'), prev_img_cv2, depth, rot_mat, translate_xyz, anim_args, keys, frame_idx)

    torch.cuda.empty_cache()
    
    return result, depth_used

def transform_image_3d_switcher(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx):
    if anim_args.depth_algorithm.lower() in ['midas+adabins (old)', 'zoe+adabins (old)']:
        return transform_image_3d_legacy(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx), depth_tensor
    else:
        return transform_image_3d_new(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx)

def transform_image_3d_legacy(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion 
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    if anim_args.aspect_ratio_use_old_formula:
        aspect_ratio = float(w)/float(h)
    else:
        aspect_ratio = keys.aspect_ratio_series[frame_idx]
    
    near = keys.near_series[frame_idx]
    far = keys.far_series[frame_idx]
    fov_deg = keys.fov_series[frame_idx]
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    if depth_tensor is None:
        z = torch.ones_like(x)
    else:
        z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode=anim_args.sampling_mode, 
        padding_mode=anim_args.padding_mode, 
        align_corners=False
    )

    # convert back to cv2 style numpy array
    return process_image(new_image)

def transform_image_3d_new(device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, keys, frame_idx):
    '''
    originally an adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion
    modified by reallybigname to control various incoming tensors, equalize, auto-contrast
    '''
    # get projection keys
    near = keys.near_series[frame_idx]
    far = keys.far_series[frame_idx]
    fov_deg = keys.fov_series[frame_idx]

    # this will become the returned depth, if depth is used
    d = None

    # midas starts with a negative/positive range
    depth_config_midas = {'depth': 1, 'offset': -2, 'invert_before': False, 'invert_after': True}
    depth_config_ones = {'depth': 1, 'offset': 1, 'invert_before': False, 'invert_after': False}
    if anim_args.depth_algorithm.lower().startswith('midas'): # 'Midas-3-Hybrid' or 'Midas-3.1-BeitLarge'
        depth_config = depth_config_midas
    elif anim_args.depth_algorithm.lower() == "adabins":
        depth_config = depth_config_ones
    elif anim_args.depth_algorithm.lower() == "leres":
        depth_config = depth_config_ones
    elif anim_args.depth_algorithm.lower() == "zoe":
        depth_config = depth_config_ones
    else:
        raise Exception(f"Unknown depth_algorithm passed to transform_image_3d function: {anim_args.depth_algorithm}")

    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    # depth stretching aspect ratio (has nothing to do with image dimensions - which is why the old formula was flawed)
    aspect_ratio = float(w)/float(h) if anim_args.aspect_ratio_use_old_formula else keys.aspect_ratio_series[frame_idx]
    
    # get perspective cams old (still) and new (transformed)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # make xy meshgrid - range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))

    # test tensor for validity (some are corrupted for some reason)
    depth_tensor_invalid = depth_tensor is None or torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any() or depth_tensor.min() == depth_tensor.max()

    # if depth_tensor is not None:
    #     debug_print(f"Depth_T.min: {depth_tensor.min()}, Depth_T.max: {depth_tensor.max()}")

    # if invalid, create flat z for this frame
    if depth_tensor_invalid:
        # if none, then 3D depth is turned off, so no warning is needed.
        if depth_tensor is not None:
            print("Depth tensor invalid. Generating a Flat depth for this frame.")
        # create flat depth
        z = torch.ones_like(x)
    # create z from depth tensor
    else:
        # prepare tensor between 0 and 1 with normalization
        d = depth_normalize(depth_tensor)

        # for inversion at this stage after normalization
        if depth_config['invert_before']: d = depth_invert(d)

        # equalization (makes motion more continuous rather than slowing down when close to stuff, speed up when not)
        d = depth_equalization(d)

        # auto-contrast to enhance depth
        d = depth_auto_contrast(d)
        
        # Rescale 0-1 depth_tensor to depth_config['depth'] and offset with depth_config['offset']
        # depth 2 and offset -1 would be from -1 to +1 | depth 2 and offset -2 would be from -2 to 0
        d *= depth_config['depth']
        d += depth_config['offset']

        # for inversion at this stage after everything else
        if depth_config['invert_after']: d *= -1 # depth_invert(d)

        # console reporting of depth normalization, min, max, diff
        # will *only* print to console if Dev mode is enabled in general settings of Deforum
        txt_depth_min, txt_depth_max = '{:.2f}'.format(float(depth_tensor.min())), '{:.2f}'.format(float(depth_tensor.max()))
        diff = '{:.2f}'.format(float(depth_tensor.max()) - float(depth_tensor.min()))
        console_txt = f"\033[36mDepth normalized to {d.min()}/{d.max()} from"
        debug_print(f"{console_txt} {txt_depth_min}/{txt_depth_max} diff {diff}\033[0m") 

        # add z from depth
        z = torch.as_tensor(d, dtype=torch.float32, device=device)

    # calculate offset_xy
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)

    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    # do the hyperdimensional remap
    image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.unsqueeze(0),
        offset_coords_2d, 
        mode=anim_args.sampling_mode, 
        padding_mode=anim_args.padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array, return processed depth
    return process_image(new_image), d

def depth_normalize(depth_tensor=None):
    if depth_tensor is None:
        raise ValueError("depth_tensor is None")

    depth_range = depth_tensor.max() - depth_tensor.min()

    # Handling the edge case where all depth values are the same
    if depth_range == 0:
        return torch.zeros_like(depth_tensor)

    # Normalize
    depth_tensor = (depth_tensor - depth_tensor.min()) / depth_range

    return depth_tensor

def depth_equalization(depth_tensor, min_val=0, max_val=1):
    """
    Perform histogram equalization on a single-channel depth tensor.
    Args:
    depth_tensor (torch.Tensor): A 2D depth tensor (H, W).
    Returns:
    torch.Tensor: Equalized depth tensor (2D).
    """
    # min/max inputs
    min_val = min(max(min_val, 0), 1)
    max_val = min(max(max_val, 0), 1)

    # Convert the depth tensor to a NumPy array for processing
    depth_array = depth_tensor.cpu().numpy()

    # Calculate the number of bins based on the specified range and round to the nearest integer
    num_bins = int(1024 * (max_val - min_val))

    # Ensure a minimum of 1 bin
    num_bins = max(num_bins, 1)

    # Calculate the histogram of the depth values using a specified number of bins
    # Increase the number of bins for higher precision depth tensors
    hist, bin_edges = np.histogram(depth_array, bins=num_bins, range=(min_val, max_val))

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()

    # Normalize the CDF so that the maximum value is 1
    cdf = cdf / float(cdf[-1])

    # Perform histogram equalization by mapping the original depth values to the CDF values
    equalized_depth_array = np.interp(depth_array, bin_edges[:-1], cdf)

    # Convert the equalized depth array back to a PyTorch tensor and return it
    equalized_depth_tensor = torch.from_numpy(equalized_depth_array).to(depth_tensor.device).type(depth_tensor.dtype)

    return equalized_depth_tensor

def depth_auto_contrast(depth_tensor, min_val=0, max_val=1):
    """
    Perform auto contrast on a single-channel depth tensor.
    Args:
    depth_tensor (torch.Tensor): A normalized 2D depth tensor (H, W).
    min (float): The minimum value of the desired range. (0-1)
    max (float): The maximum value of the desired range. (0-1)
    Returns:
    torch.Tensor: Auto-contrasted depth tensor (2D).
    """
    # minmax the input min and max!
    min_val = minmax(min_val, 0, 1)
    max_val = minmax(max_val, 0, 1)

    # Perform auto contrast by mapping the normalized depth values to the specified range
    auto_contrasted_depth_tensor = depth_tensor * (max_val - min_val) + min_val

    return auto_contrasted_depth_tensor

def minmax(val, min_val, max_val):
    return min(max(val, min_val), max_val)

def depth_invert(depth_tensor):
    max_depth = torch.max(depth_tensor)
    inverted_depth_tensor = max_depth - depth_tensor
    return inverted_depth_tensor

def combine_cv2_with_p3d_perspective(M, translation_scale, R_p3d, translate, translation_z_estimate, device):
    # translate comes in as a list
    # r_p3d comes in as a torch tensor

    # Extract the rotation and translation components from the cv2 matrix
    rot_list_cv2, translation_z_cv2 = get_rot_trans_z_from_cv2_perspective(M)

    # convertr extracted angles of rotation from cv2 matrix to pytoch matrix like original rot_mat
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rot_list_cv2, device=device), "XYZ").unsqueeze(0)

    # extract translation from cv2 matrix, and add translation_z from cv2 rotation/scaling matrix extraction     
    translate_cv2 = [M[0][2], M[1][2], M[2][2] + translation_z_cv2] # + translation_z_estimate]

    # scale cv2 translation to match then combine tranblation from cv2 with the translation list
    scaled_translation_cv2 = [t * translation_scale for t in translate_cv2]

    # Combine the rotation and translation extracted from cv2 matrix with the input and return
    combined_R = torch.matmul(R_p3d, rot_mat)
    combined_T = [t1 + t2 for t1, t2 in zip(scaled_translation_cv2, translate)]

    return combined_R, combined_T

def get_rot_trans_z_from_cv2_perspective(matrix):
    if matrix.shape == (3, 3):
        # Clamping matrix elements to prevent errors in euler calcs
        matrix = np.clip(matrix, -1, 1)        
        
        # Extract the rotation elements
        r11 = matrix[0, 0]
        r12 = matrix[0, 1]
        r21 = matrix[1, 0]
        r22 = matrix[1, 1]

        # Calculate the scaling factor (average of the diagonal elements)
        scaling_factor = (r11 + r22) / 2.0

        # Approximate translation_z as the scaling factor
        translation_z = scaling_factor

        # Calculate the rotation angles as Euler angles (x, y, z)
        x_euler = math.atan2(r21, r22)
        y_euler = math.asin(-r12)
        z_euler = math.atan2(r11, r12)

        # Convert the rotation euler angles to radians
        x_rad = math.radians(x_euler)
        y_rad = math.radians(y_euler)
        z_rad = math.radians(z_euler)
        radians_list = [x_rad, y_rad, z_rad]

        return radians_list, translation_z
    else:
        return None

def border_to_cv2_handle(border):
    if border == 'wrap':
        r = cv2.BORDER_WRAP
    elif border == 'replicate':
        r = cv2.BORDER_REPLICATE
    elif border == 'reflect':
        r = cv2.BORDER_REFLECT_101
    else: # BORDER_DEFAULT is the same as BORDER_REFLECT_101
        r = cv2.BORDER_DEFAULT
    return r

def process_image(new_image, output_dtype=np.uint8, clamp_values=True):
    """
    Process a PyTorch image tensor and convert it into a NumPy array.

    Args:
    new_image (torch.Tensor): A PyTorch tensor representing an image.
    output_dtype (type, optional): Desired NumPy data type for the output image (e.g., np.uint8, np.uint16, np.float32). Default is np.uint8.
    clamp_values (bool, optional): Whether to clamp the values. Useful for integer types. Default is True.

    Returns:
    numpy.ndarray: Processed image as a NumPy array.
    """
    # Squeeze and rearrange the tensor from 'C x H x W' to 'H x W x C'
    image_processed = rearrange(new_image.squeeze(), 'c h w -> h w c').cpu().numpy()

    # Clamping values if needed
    if clamp_values and np.issubdtype(output_dtype, np.integer):
        min_val, max_val = np.iinfo(output_dtype).min, np.iinfo(output_dtype).max
        image_processed = np.clip(image_processed, min_val, max_val)

    # Convert to desired data type
    image_processed = image_processed.astype(output_dtype)

    return image_processed

# UNUSED BELOW --------------------------

def depth_contrast(depth_tensor, factor):
    # Adjust the contrast of the normalized depth while preserving the normalized scale
    dt = depth_tensor + (depth_tensor - 0.5) * (factor - 1.0)

    # Ensure values stay within the [0, 1] range
    dt = torch.clamp(dt, 0, 1)

    return dt

def depth_shift_distribution(depth_tensor, control=0.0):
    # Get the min and max of the depth tensor
    min_val = depth_tensor.min()
    max_val = depth_tensor.max()

    # Shift the distribution of the depth tensor
    if control < 0:
        depth_tensor = max_val - torch.pow(max_val - depth_tensor, 1 / (1 - control))
    else:
        depth_tensor = min_val + torch.pow(depth_tensor - min_val, 1 / (1 + control))

    return depth_tensor

def depth_stretch_distribution(depth_tensor, control=0.0):
    # Get the min and max of the depth tensor
    min_val = depth_tensor.min()
    max_val = depth_tensor.max()

    # Adjust the distribution of the depth tensor
    if control < 0:
        depth_tensor = ((max_val + min_val) / 2) * (1 - torch.cos((depth_tensor - min_val) / (max_val - min_val) * math.pi))
    else:
        depth_tensor = min_val + (depth_tensor - min_val) ** (1 / (control + 1))

    return depth_tensor
