import os
import cv2
import gc
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from .video_audio_utilities import get_frame_name
from .load_images import load_image

def do_overlay_mask(args, anim_args, img, frame_idx, inputframes_path, maskframes_path, video_mask_override=None, mask_override=None, invert_override=None, is_bgr_array=True):
    # this function does overlay_mask WITHOUT using automatic1111, because these overlays work during cadence!
    # called twice in render 2d/3d, once in cadence, once outside if cadence is 1
    # - "mask_override" is an optional parameter to send the mask directly to this function, whether for use_mask_video or for use_mask
    # - "is_bgr_array" allows for BGR np.array input/output during cadence (normally PIL outside of cadence)

    if is_bgr_array: img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if anim_args.use_mask_video or video_mask_override is not None:
        # get video mask override or open video mask frame from file
        if video_mask_override is not None:
            current_mask = Image.fromarray(cv2.cvtColor(video_mask_override, cv2.COLOR_BGR2RGB))
        else:
            current_mask = Image.open(os.path.join(maskframes_path, get_frame_name(anim_args.video_mask_path) + f"{frame_idx:09}.jpg"))
        # current video frame from inputframes
        current_frame = Image.open(os.path.join(inputframes_path, get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
    elif args.use_mask or mask_override is not None:
        # get mask override, else load mask or mask file
        if mask_override is not None:
            current_mask = Image.fromarray(cv2.cvtColor(mask_override, cv2.COLOR_BGR2RGB))
        else:
            current_mask = args.mask_image if args.mask_image is not None else load_image(args.mask_file)
        current_frame = img if args.init_image is None else load_image(args.init_image)

    # resize
    dimensions = (args.W, args.H)
    current_mask = current_mask.resize(dimensions, Image.LANCZOS)
    current_frame = current_frame.resize(dimensions, Image.LANCZOS)

    # convert to grayscale, do contrast, brightness, overlay blur, inversion
    current_mask = ImageOps.grayscale(current_mask)
    
    if mask_override is not None or video_mask_override is not None:
        if invert_override:
            current_mask = ImageOps.invert(current_mask)
    else:
        if float(args.mask_contrast_adjust) != 1:
            current_mask = ImageEnhance.Contrast(current_mask).enhance(float(args.mask_contrast_adjust))
        if float(args.mask_brightness_adjust) != 1:
            current_mask = ImageEnhance.Brightness(current_mask).enhance(float(args.mask_brightness_adjust))
        if float(args.mask_overlay_blur) > 0:
            current_mask = current_mask.filter(ImageFilter.GaussianBlur(radius = float(args.mask_overlay_blur)))
        if args.invert_mask:
            current_mask = ImageOps.invert(current_mask)

    # do composite
    img = Image.composite(img, current_frame, current_mask)

    # saving of overlay masks that come from hybrid video if save extra frames in on
    if (anim_args.hybrid_composite != 'None' or anim_args.hybrid_motion != 'None') and anim_args.hybrid_comp_save_extra_frames:
        hybridframes_path = os.path.join(args.outdir, 'hybridframes')
        img.save(os.path.join(hybridframes_path, f"overlay_{frame_idx:09}.jpg"))

    if is_bgr_array: img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    del current_mask, current_frame
    gc.collect()

    return img