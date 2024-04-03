from .hybrid_flow import get_flow_from_images, image_transform_optical_flow, stabilize_flow_motion, stabilize_flow_rotation
from .updown_scale import updown_scale_whatever
from .general_utils import debug_print

def do_temporal_flow(img_prev, img, flow_method, idx, raft_model, flow_factor=1.0, motion_stabilizer_factor=1.0, rotation_stabilizer_factor=1.0, return_target='Image', target_frame=1, updown_scale=1, suppress_console=False, return_image=True, return_flow=True):
    ''' Temporal flow - determine flow from a previous image to current image, apply flow to current image.
    '''
    # upscale images and flow will be upscaled when created
    if updown_scale > 1:
        img_prev, _ = updown_scale_whatever(img_prev, scale=updown_scale)
        img, img_shape = updown_scale_whatever(img, scale=updown_scale)

    # get flow from img to img_prev previous images
    flow = get_flow_from_images(img_prev, img, flow_method, raft_model)

    if motion_stabilizer_factor != 0:
        flow = stabilize_flow_motion(flow, motion_stabilizer_factor)
    if rotation_stabilizer_factor != 0:
        flow = stabilize_flow_rotation(flow, rotation_stabilizer_factor)

    # if returning an image, determine which image to affect and apply optical flow to the return_image
    if return_image:
        # warp image with flow factor 1, since flow factor was already applied above
        return_target_image = img if return_target == 'Image' else img_prev
        return_target_image = image_transform_optical_flow(return_target_image, flow, flow_factor)

        # downscale (don't downscale flow, since it is reused as-is)
        if updown_scale > 1:
            return_target_image, _ = updown_scale_whatever(return_target_image, shape=img_shape)
    else:
        # if only returning flow, apply flow factor here
        if return_flow:
            flow *= flow_factor

    # console reporting
    debug_print(f"Temporal {flow_method} flow captured target {target_frame} frames back {idx-target_frame}→{idx} ", not suppress_console)
    if motion_stabilizer_factor != 0 or rotation_stabilizer_factor != 0:
        debug_print(f"Temporal flow stabilization factors: motion: {motion_stabilizer_factor} | rotation: {rotation_stabilizer_factor}", not suppress_console)
    if return_image:
        image_or_target_image = 'image' if return_target == 'Image' else 'target image'
        debug_print(f"Temporal flow applied to {image_or_target_image} on frame {idx} at factor {flow_factor} ", not suppress_console)

    # flow is potentially upscaled
    if updown_scale > 1 and return_flow:
        flow = updown_scale_whatever(flow, shape=img_shape)
        
    # selective returns
    if return_image and return_flow:
        return return_target_image, flow
    elif not return_image:
        return flow
    else:
        return return_target_image
