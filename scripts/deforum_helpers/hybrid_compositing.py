import os
import cv2
import numpy as np
from .video_audio_utilities import get_frame_name
from .updown_scale import updown_scale_whatever, updown_scale_to_integer
from .composite import composite_images_with_alpha, make_composite, composite_images_with_mask
from .conform import conform_images
from .load_images import load_image

def create_frame_path(base_path, frame_name, idx_ext, prefix=""):
    path_string = f"{frame_name}{idx_ext}"
    prefix_string = "" if prefix == "" else f"{prefix}_"
    return os.path.join(base_path, prefix_string + path_string)

def hybrid_composite(args, anim_args, frame_idx, img, depth_model, hybrid_comp_schedules, inputframes_path, hybridframes_path, root, conform=None, raft_model=None, return_iterations=False):
    src_path = anim_args.video_init_path
    mask_type = anim_args.hybrid_comp_mask_type
    comp_type = anim_args.hybrid_comp_type
    updown_scale = updown_scale_to_integer(anim_args.updown_scale)
    mask_alpha = float(hybrid_comp_schedules['mask_alpha'])
    hybrid_mask = None
    hybrid_overlay = None
    conform_iterations = anim_args.hybrid_comp_conform_iterations
    conform_method = anim_args.hybrid_comp_conform_method

    # upscaling before other functions are called
    if updown_scale > 1:
        img, img_shape = updown_scale_whatever(img, scale=updown_scale)
    
    # if updownscaling, this picks up dimensions from scaled img when resizing video
    dimensions = (img.shape[1], img.shape[0])

    # frame paths for saving and for saving extra images
    idx_ext = f"{frame_idx:09}.jpg"
    frame_name = get_frame_name(src_path)
    frame_types = ["", "mask", "mask_color", "overlay", "comp", "mix", "conform"]
    frame_paths = {frame_type: create_frame_path(hybridframes_path, frame_name, idx_ext, prefix=frame_type) for frame_type in frame_types}
    frame_paths.update({'video': os.path.join(inputframes_path, frame_name + idx_ext)})

    # determine init image or video to work with and resize video image to match dimensions (updownscaled or not)
    video_image = load_image(args.init_image, return_np_bgr=True) if anim_args.hybrid_use_init_image else cv2.imread(frame_paths['video'])
    video_image = cv2.resize(video_image, dimensions, interpolation=cv2.INTER_LANCZOS4)

    # force image to video shape or video to image shape for compositing
    if anim_args.diffusion_cadence == 1 and anim_args.hybrid_comp_conform_method != 'None' and conform is not None:
        # call conform function and get conform return var which contains the warped images, and optionally the iteration gif
        conform_dict = conform_images(img, video_image, conform, flow_method=conform_method, raft_model=raft_model, iterations=conform_iterations, return_iterations=return_iterations)
        # write conformed images back to vars
        img =         conform_dict['image1']
        video_image = conform_dict['image2']

        # optional saving of image strips of conform operations
        if return_iterations:
            cv2.imwrite(frame_paths['conform'], conform_dict['strip1'])

    # create mask
    if mask_type in make_composite()['grayscale']:
        # still full color at this point when mask creation happens
        hybrid_mask = make_composite(mask_type, video_image, img, mask_alpha)
    elif mask_type in ['Depth', 'Video Depth']:
        # depth in the image or in the video image
        hybrid_mask_tensor = depth_model.predict(img if mask_type == 'Depth' else video_image, anim_args.midas_weight, root.half_precision)
        hybrid_mask = depth_model.to_image(hybrid_mask_tensor, output_format='opencv')    
    # if no mask, the hybrid composite is the video image
    if hybrid_mask is None:
        hybrid_comp = video_image
    elif mask_type != "None":
        # optionally save color mask before grayscale
        if anim_args.hybrid_comp_save_extra_frames:
            cv2.imwrite(frame_paths['mask_color'], hybrid_mask.astype(np.uint8))

        # make grayscale
        hybrid_mask = cv2.cvtColor(np.array(hybrid_mask), cv2.COLOR_BGR2GRAY)

        # invert mask
        if anim_args.hybrid_comp_mask_inverse:
            hybrid_mask = cv2.bitwise_not(hybrid_mask.astype(np.uint8))

        # equalization before
        if anim_args.hybrid_comp_mask_equalize in ['Before', 'Both']:
            hybrid_mask = cv2.equalizeHist(hybrid_mask.astype(np.uint8))

        # contrast
        hybrid_mask = cv2.convertScaleAbs(hybrid_mask.astype(np.uint8), alpha=hybrid_comp_schedules['mask_contrast'])

        # auto contrast with cutoffs lo/hi
        if anim_args.hybrid_comp_mask_auto_contrast:
            hybrid_mask = autocontrast_grayscale(np.array(hybrid_mask.astype(np.uint8)), hybrid_comp_schedules['mask_auto_contrast_cutoff_low'], hybrid_comp_schedules['mask_auto_contrast_cutoff_high'])
            
        # optional mask save
        if anim_args.hybrid_comp_save_extra_frames:
            cv2.imwrite(frame_paths['mask'], hybrid_mask.astype(np.uint8))

        # equalization after
        if anim_args.hybrid_comp_mask_equalize in ['After', 'Both']:
            hybrid_mask = cv2.equalizeHist(hybrid_mask.astype(np.uint8))

        # make hybrid composite of image and video using hybrid_mask ('compositing mask filter')
        hybrid_comp = composite_images_with_mask(video_image, img, hybrid_mask.astype(np.uint8))

        # optionally save composited image
        if anim_args.hybrid_comp_save_extra_frames:
            cv2.imwrite(frame_paths['comp'], hybrid_comp)

    # compositing filter (not the 'compositing mask filter' - the 'compositing filter')
    # final mix of hybrid composite at hybrid comp alpha (1 if full composite)
    composite_alpha = hybrid_comp_schedules['alpha']
    hybrid_comp_args = (hybrid_comp, img, composite_alpha)
    if comp_type == 'None':
        mixed = composite_images_with_alpha(*hybrid_comp_args)
    else:
        mixed = make_composite(comp_type, *hybrid_comp_args)

    # optionally save mix of image with composited image (that gets sent back to generation)
    if anim_args.hybrid_comp_save_extra_frames:
        cv2.imwrite(frame_paths['mix'], mixed)

    # hybrid mask as overlay
    if anim_args.hybrid_comp_mask_do_overlay_mask != "None" and hybrid_mask is not None:
        hybrid_overlay = np.copy(hybrid_mask.astype(np.uint8))

        # optionally save hybrid overlay
        if anim_args.hybrid_comp_save_extra_frames:
            cv2.imwrite(frame_paths['overlay'], hybrid_overlay)

    # upscaling before other functions are called
    if updown_scale > 1:
        mixed, _ = updown_scale_whatever(mixed, shape=img_shape)
   
    return mixed.astype(np.uint8), hybrid_overlay

def autocontrast_grayscale(image, low_cutoff=0.0, high_cutoff=1.0):
    # Convert the cutoffs from proportions to pixel intensity values
    low_cutoff *= 255
    high_cutoff *= 255

    # Perform autocontrast on a grayscale np array image.
    # Scale the image so that the minimum value is 0 and the maximum value is 255
    image = 255 * (image - low_cutoff) / (high_cutoff - low_cutoff)

    # Clip values that fall outside the range [0, 255]
    image = np.clip(image, 0, 255)

    return image
