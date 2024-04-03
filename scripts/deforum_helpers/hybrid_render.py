from .hybrid_ransac import get_matrix_for_hybrid_motion, image_transform_ransac, get_default_matrix
from .hybrid_flow import get_flow_for_hybrid_motion, image_transform_optical_flow, get_default_flow
from .updown_scale import updown_scale_to_integer
from .general_utils import debug_print

def get_hybrid_motion_vars(anim_args, frame_idx, dimensions, reverse=False):
    ''' shared vars for hybrid_motion & hybrid_motion_cadence '''

    # anim_args translated to shorter names
    motion = anim_args.hybrid_motion
    motion_behavior = anim_args.hybrid_motion_behavior
    use_prev_img = anim_args.hybrid_motion_use_prev_img
    flow_method = anim_args.hybrid_flow_method
    flow_consistency = anim_args.hybrid_flow_consistency
    consistency_blur = anim_args.hybrid_consistency_blur
    save_extra_frames = anim_args.hybrid_comp_save_extra_frames
    max_frames = anim_args.max_frames

    # before generation gets flow from last frame to current frame (original behavior)
    # after generation gets flow from current frame to next frame
    # all cadence frames are after generation
    motion_before = motion_behavior == 'Before Generation'

    # only do motion if flow doesn't need to read files before first idx or after last index
    do_motion = anim_args.animation_mode in ['2D', '3D'] and ((motion_before and frame_idx > 0) or (not motion_before and frame_idx < max_frames))
  
    # determine last index from current frame index
    last_idx = frame_idx - (1 if motion_before else 0)
    # this index is 1 higher than last_idx
    this_idx = last_idx + 1

    if reverse:
        this_idx, last_idx = last_idx, this_idx

    return motion, use_prev_img, flow_method, flow_consistency, consistency_blur, save_extra_frames, do_motion, last_idx, this_idx, dimensions

def hybrid_motion(frame_idx, img, prev_img, prev_motion, args, anim_args, inputfiles, hybridframes_path, hybrid_flow_factor, raft_model, matrix_flow=None, suppress_console=False, reverse=False, image_return=True):
    ''' function for hybrid motion - used by cadence too '''

    updown_scale = updown_scale_to_integer(anim_args.updown_scale)
    
    # IN PROGRESS - experimental vars not in ui
    anchoring = False
    use_img_reference = False

    img_reference = img if use_img_reference else None

    # get anim_arg vars
    hybrid_motion_type, use_prev_img, flow_method, flow_consistency, consistency_blur, \
    save_extra_frames, do_motion, last_idx, this_idx, dimensions = get_hybrid_motion_vars(anim_args, frame_idx, (args.W, args.H), reverse=reverse)

    # do motion if okay
    if do_motion:
        # calculate optical flow or matrix as 'motion', warp img based on motion
        if hybrid_motion_type in ['Optical Flow', 'Matrix Flow']:
            motion = get_flow_for_hybrid_motion(last_idx, this_idx, dimensions, inputfiles, hybridframes_path, prev_motion, hybrid_motion_type, flow_method, anim_args.animation_mode, raft_model,
                                                None, None, use_prev_img, prev_img, flow_consistency, consistency_blur, save_extra_frames, updown_scale, matrix_flow=matrix_flow,
                                                anchoring=anchoring, img=img_reference, suppress_console=suppress_console)
            if image_return:
                debug_print(f"Applying optical flow at {hybrid_flow_factor} to frame {this_idx}")
                img = image_transform_optical_flow(img, motion, hybrid_flow_factor)
        else: # hybrid_motion_type in ['Affine', 'Perspective']
            motion = get_matrix_for_hybrid_motion(last_idx, this_idx, dimensions, inputfiles, hybrid_motion_type, use_prev_img, prev_img, suppress_console=suppress_console)
            debug_print(f"Applying {hybrid_motion_type} matrix to frame {this_idx}")
            if image_return:
                img = image_transform_ransac(img, motion, hybrid_motion_type)
    else: # aborted flow with defaults and no warp
        motion = get_default_motion(hybrid_motion_type, dimensions)

    if image_return:
        # return warped image and motion used
        return img, motion
    else:
        return motion

def get_default_motion(motion, dimensions=None):
    if motion in ['Optical Flow', 'Matrix Flow']:
        motion = get_default_flow(dimensions)
    else:
        motion = get_default_matrix(motion)
    
    return motion

def get_hybrid_matrix_flow(frame_idx, max_frames, last_image, dimensions, use_prev_img, hybrid_motion_behavior, hybrid_flow_method, mode, inputfiles, hybridframes_path, prev_flow, raft_model, depth_model, depth, suppress_console=False):
    ''' function for getting Affine or Perspctive matrix from hybrid motion for use in 2D/3D animation routines '''

    # If user in wrong mode it doesn't matter, they don't show for Hybrid Flow, so force mode to 2D/Affine or 3D/Perspective
    ransac_motion = 'Affine' if mode == '2D' else 'Perspective'
    method = hybrid_flow_method

    # get args, anim_arg vars
    motion_before = hybrid_motion_behavior == 'Before Generation'    
    do_motion = mode in ['2D', '3D'] and ((motion_before and frame_idx > 0) or (not motion_before and frame_idx < max_frames))

    # last index offset determined by motion behavior
    last_idx = frame_idx
    if motion_before:
        last_idx -= 1
    this_idx = last_idx + 1

    if do_motion:
        matrix_motion_args = (last_idx, this_idx, dimensions, inputfiles, ransac_motion)
        matrix_motion_kwargs = {'suppress_console': suppress_console}
        matrix = get_matrix_for_hybrid_motion(*matrix_motion_args, **matrix_motion_kwargs)

        flow_motion_args = (last_idx, this_idx, dimensions, inputfiles, hybridframes_path, prev_flow, 'Matrix Flow', method, mode, raft_model, depth_model, depth)
        flow_motion_kwargs = {'use_prev_img': use_prev_img, 'prev_img': last_image, 'suppress_console': suppress_console}
        flow = get_flow_for_hybrid_motion(*flow_motion_args, **flow_motion_kwargs)

    else: # aborted flow with defaults and no warp
        matrix = get_default_motion(ransac_motion, dimensions)
        flow = get_default_motion('Matrix Flow', dimensions)
   
    # return matrix for 2D or 3D, and a flow
    return (matrix, flow)

