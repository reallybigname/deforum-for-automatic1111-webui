import numpy as np
import cv2
from tqdm import tqdm
from .morphological import image_morphological_flow_transform, morpho_get_random_op, morpho_get_random_kernel, extract_morpho_schedule_item
from .updown_scale import updown_scale_whatever, updown_scale_to_integer
from .cadence import set_range_get_total, better_range, reverse_index
from .hybrid_flow import image_transform_optical_flow, get_default_flow
from .general_utils import debug_print

# modifies turbo object for morphological transformation
def cadence_morpho(turbo, anim_args, raft_model):   
    behavior = anim_args.morpho_cadence_behavior
    min_conditions_met = anim_args.morpho_flow != 'None' and behavior != 'None'
    forward = behavior == "Forward" # forward False means behavior is 'Bounce' mode
    direct_mode = anim_args.morpho_flow == 'No Flow (Direct)'
    
    if min_conditions_met:
        # used to accumulate flows and apply them in order again and again as we advance through idx
        updown_scale = updown_scale_to_integer(anim_args.updown_scale)

        morpho = None    
        morpho_kwargs = {'return_image': True if direct_mode else False, 'suppress_console': True}

        # establish range for loop, get total for progress bar
        start_range = turbo['idx_start']
        end_range = turbo['idx_end']
        end_modifier = 0 if forward or direct_mode else -1 # Bounce mode doesn't affect the last frame, so end range gets shifted here
        r, rt = set_range_get_total(start_range, end_range + end_modifier, 1, False)

        # initialize morpho var array for direct mode, or set up default flows for prev/next stored flows
        if direct_mode:
            morpho_vars = []
        else:
            dimensions = (turbo[r[0]]['prev'].shape[1], turbo[r[0]]['prev'].shape[0]) 
            morpho_prev_flow_store = get_default_flow(dimensions)
            morpho_next_flow_store = get_default_flow(dimensions)

        pbar = tqdm(total=rt)
        for idx in better_range(r[0], r[1], r[2], r[3]):
            pbar.set_description(f"Cadence morphological flow for frames {r[0]}→{r[1]-1} ")
            pbar.update(1)

            # get cadence vars for this index, next is reverse indexed if not forward mode
            t = turbo[idx]
            rdx = reverse_index(idx, r[0], r[1])
            prev, next, ckey = t['prev'], t['next'] if forward else turbo[rdx]['next'], t['ckey']

            # get morpho first thing - op and kernel remains the same through cadence cycle (even if random), but iterations can change
            if morpho is None:
                # morpho = ckey['morpho']
                # morpho_ok = morpho.split("|") # split to [op, kernel]
                morpho_o, morpho_k = extract_morpho_schedule_item(ckey['morpho'])
                if morpho_o == 'random': morpho_o = morpho_get_random_op()
                if morpho_k == 'random': morpho_k = morpho_get_random_kernel()
                morpho = f"{morpho_o}|{morpho_k}"

            if ckey['morpho_iterations'] != 0 and (direct_mode or (not direct_mode and ckey['morpho_flow_factor'] != 0)):
                morpho_args = (anim_args.morpho_image_type, anim_args.morpho_bitmap_threshold, anim_args.morpho_flow, morpho, ckey['morpho_iterations'], idx, raft_model)

                # tween['image'] contains cadence easing locked to 'prime' linear master | if tween is alpha, neewt is beta (1-tween)
                tween_image = ckey['tween']['image']
                neewt_image = 1 - tween_image

                # upscale (does nothing if scale is 1)
                if updown_scale > 1:
                    prev, prev_shape = updown_scale_whatever(prev, scale=updown_scale)
                    next, next_shape = updown_scale_whatever(next, scale=updown_scale)

                # returns image directly without using flow, in whatever color morpho image type is set to
                if direct_mode:
                    # catch-up routine does all transforms so far in this cadence cycle before doing this frame's
                    for the_args in morpho_vars:
                        prev = image_morphological_flow_transform(prev, *the_args, **morpho_kwargs)
                        next = image_morphological_flow_transform(next, *the_args, **morpho_kwargs)

                    # do this frame's morphological operations directly and store in a dict for catch-up routine above
                    prev = image_morphological_flow_transform(prev, *morpho_args, **morpho_kwargs)
                    next = image_morphological_flow_transform(next, *morpho_args, **morpho_kwargs)
                    morpho_vars.append(morpho_args)
                else:
                    # get morpho flows only (no return image)
                    morpho_prev_flow = image_morphological_flow_transform(prev, *morpho_args, **morpho_kwargs)
                    morpho_next_flow = image_morphological_flow_transform(next, *morpho_args, **morpho_kwargs)
                    
                    # tween morpho flows for each mode and apply flow factor to flow directly (rather than at remap time)
                    morpho_prev_flow *= neewt_image if forward else tween_image
                    morpho_next_flow *= tween_image
                    morpho_prev_flow *= ckey['morpho_flow_factor']
                    morpho_next_flow *= ckey['morpho_flow_factor']

                    # store up the remapped flow every frame
                    morpho_prev_flow_store = cv2.add(morpho_prev_flow_store, morpho_prev_flow)
                    morpho_next_flow_store = cv2.add(morpho_next_flow_store, morpho_next_flow)

                    # do this frames flow (contains all previous flows from this cadence too)
                    prev = image_transform_optical_flow(prev, morpho_prev_flow_store)
                    next = image_transform_optical_flow(next, morpho_next_flow_store)

                # downscale
                if updown_scale > 1:
                    prev, _ = updown_scale_whatever(prev, shape=prev_shape)
                    next, _ = updown_scale_whatever(next, shape=next_shape)
                
                # update turbo dict
                turbo[idx]['prev'] = np.copy(prev)
                turbo[idx if forward else rdx]['next'] = np.copy(next)
        pbar.close()

    return turbo
