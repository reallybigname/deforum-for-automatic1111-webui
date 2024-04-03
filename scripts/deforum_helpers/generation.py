import cv2
import random
import numpy as np
from .colors import color_coherence
from .hybrid_flow import get_flow_from_images, image_transform_optical_flow
from .generate import generate
from .updown_scale import updown_scale_to_integer
from .general_utils import debug_print

# Redo generation, before generation
def redo_generation(bgr_image, cc_sample, cc_alpha, args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, redo_for_cc_before_gen=False, suppress_console=False):
    # if redo_for_cc_before_gen mode, count is 1
    count = 1 if redo_for_cc_before_gen else int(anim_args.diffusion_redo)

    # set up color coherence
    cc = color_coherence()

    # redo after generation for color coherence before generation. only cc before generation, not cc after like normal redo
    # this is the first time it could be color matched, since cc_sample available on frame 0, but no image to match against until now
    if redo_for_cc_before_gen and anim_args.color_coherence != 'None':
        bgr_image = cc.maintain_colors(np.copy(bgr_image), cc_sample, cc_alpha, anim_args.color_coherence, frame, console_msg="redo generation 1st frame for before generation color coherence")
    
    # loop redo
    for n in range(0, count):
        if redo_for_cc_before_gen:
            debug_print(f"Redo of 1st generation to fulfill color cohrence to {anim_args.color_coherence_source} before generation", not suppress_console)
        else:
            debug_print(f"Redo generation {n + 1} of {int(anim_args.diffusion_redo)} before final generation", not suppress_console)

            # uses random seed for extra generation if not on special redo_for_cc_before_gen
            args.seed = random.randint(0, 2 ** 32 - 1)

        # generate and convert to BGR np array
        rgb = generate(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name)
        bgr_image = cv2.cvtColor(np.array(np.copy(rgb)), cv2.COLOR_RGB2BGR)

        # BGR color match on last one only
        if n == count-1 and not redo_for_cc_before_gen and anim_args.color_coherence != 'None':
            bgr_image = cc.maintain_colors(np.copy(bgr_image), cc_sample, cc_alpha, anim_args.color_coherence, frame, console_msg="redo generation", suppress_console=suppress_console)

    return bgr_image

# Optical flow generation, happens before generation
def optical_flow_generation(bgr_image_in, cc_sample, cc_alpha, redo_flow_factor, raft_model, args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name, suppress_console=False):
    # set up color coherence
    cc = color_coherence()

    # uses random seed for extra generation
    args.seed = random.randint(0, 2 ** 32 - 1)

    # generate 
    rgb = generate(args, keys, anim_args, loop_args, controlnet_args, root, frame, sampler_name)
    
    # convert to BGR np array
    bgr_image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    
    # get flow from BGR image to new BGR image generation
    flow = get_flow_from_images(np.copy(bgr_image_in), np.copy(bgr_image), anim_args.optical_flow_redo_generation, raft_model, updown_scale=updown_scale_to_integer(anim_args.updown_scale))

    # transform new generation with the flow from old to new, and color match again
    bgr_image = image_transform_optical_flow(np.copy(bgr_image), flow, redo_flow_factor)
    
    # if there is a color coherence sample, do color matching on new image
    if cc_sample is not None and anim_args.color_coherence != 'None':
        bgr_image = cc.maintain_colors(bgr_image, cc_sample, cc_alpha, anim_args.color_coherence, frame, console_msg="opt flow generation", suppress_console=suppress_console)

    # console message
    debug_print(f"Optical flow generation applied {anim_args.optical_flow_redo_generation} flow to {frame} using extra generation", not suppress_console)

    # pass back new image, args, and root
    return bgr_image
