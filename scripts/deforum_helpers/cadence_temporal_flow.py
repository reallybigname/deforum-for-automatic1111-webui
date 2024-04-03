import numpy as np
import cv2
from tqdm import tqdm
from .temporal_flow import do_temporal_flow
from .hybrid_flow import image_transform_optical_flow, get_default_flow
from .updown_scale import updown_scale_whatever, updown_scale_to_integer
from .cadence import set_range_get_total, better_range, reverse_index
from .general_utils import debug_print

# modifies turbo object for temporal flow
def cadence_temporal_flow(turbo, anim_args, raft_model, suppress_console=False):
    start_range = turbo['idx_start']
    end_range = turbo['idx_end']
    turbo_steps = end_range - start_range
    temporal_flow_cadence_behavior = anim_args.temporal_flow_cadence_behavior
    forward = temporal_flow_cadence_behavior == 'Forward'
    conditions_met = anim_args.temporal_flow != 'None' and temporal_flow_cadence_behavior != 'None'

    if conditions_met:
        updown_scale = updown_scale_to_integer(anim_args.updown_scale)

        # establish range for loop, get total for progress bar: floor((end_range - start_range) / increment)
        end_range_modifier = 0 if forward else -1
        r, rt = set_range_get_total(start_range, end_range + end_range_modifier, 1, False)

        # set up flow storing vars, kwargs
        dimensions = (turbo[r[0]]['prev'].shape[1], turbo[r[0]]['prev'].shape[0])
        temporal_prev_flow_store = get_default_flow(dimensions)
        temporal_next_flow_store = get_default_flow(dimensions)
        temporal_kwargs = {'return_image': False, 'suppress_console': True, 'return_target': anim_args.temporal_flow_return_target}

        temporal_flow_cadence_message = "Temporal flow cadence is referring to the last turbo object for temporal flow during cadence" if 'idx_start' in turbo['turbo_last'] else \
                                        "Temporal flow cadence did not find last turbo object. Using the last image before turbo object was created instead. (1st cadence cycle?)"
        debug_print(temporal_flow_cadence_message, not suppress_console)

        pbar = tqdm(total=rt)
        for idx in better_range(r[0], r[1], r[2], r[3]):
            pbar.set_description(f"Cadence temporal flow frames {r[0]}→{r[1]-1} ")
            pbar.update(1)

            rdx = reverse_index(idx, r[0], r[1])
            # get cadence vars for this index, next is reverse indexed if not forward mode
            t = turbo[idx]
            tr = turbo[rdx]
            ckey = t['ckey']
            rckey = tr['ckey']

            if ckey['temporal_flow_factor'] != 0:
                prev, next = t['prev'], t['next'] if forward else tr['next']
                temporal_args = (anim_args.temporal_flow, idx, raft_model)

                # during cadence, temporal flow doesn't pay attention to the target frame schedule - only on actual frames
                # instead, it refers to the corresponding frame from the last turbo object (which is always stored in the current turbo object)
                if 'idx_start' in turbo['turbo_last']:
                    # establish index for images one cadence cycle ago and retrieve images
                    last_prev = turbo['turbo_last'][idx-turbo_steps]['prev']
                    last_next = turbo['turbo_last'][idx-turbo_steps]['next']
                else:
                    # if first cadence cycle, it must use the last_img (which is the frame 0 image)
                    last_prev = turbo['last_img']
                    last_next = turbo['last_img']

                # tween['image'] contains cadence easing locked to 'prime' linear master | if tween is alpha, neewt is beta (1-tween)
                tween_image = ckey['tween']['image']
                neewt_image = 1 - tween_image

                # upscale (does nothing if scale is 1)
                if updown_scale > 1:
                    last_prev, _ = updown_scale_whatever(last_prev, scale=updown_scale)
                    last_next, _ = updown_scale_whatever(last_next, scale=updown_scale)
                    prev, prev_shape = updown_scale_whatever(prev, scale=updown_scale)
                    next, next_shape = updown_scale_whatever(next, scale=updown_scale)

                # get temporal flows only (no return image)
                switching_prev_args = (last_prev, prev) #if forward else (last_prev, next)
                switching_next_args = (last_next, next) #if forward else (next, last_next)

                temporal_kwargs.update({
                    'motion_stabilizer_factor': ckey['temporal_flow_motion_stabilizer_factor'],
                    'rotation_stabilizer_factor': ckey['temporal_flow_rotation_stabilizer_factor']
                })
                temporal_prev_flow = do_temporal_flow(*switching_prev_args, *temporal_args, **temporal_kwargs)

                if not forward:
                    temporal_kwargs.update({
                        'motion_stabilizer_factor': rckey['temporal_flow_motion_stabilizer_factor'],
                        'rotation_stabilizer_factor': rckey['temporal_flow_rotation_stabilizer_factor']
                    })
                temporal_next_flow = do_temporal_flow(*switching_next_args, *temporal_args, **temporal_kwargs)

                # tween morpho flows for each mode and store for catch-up routine above
                temporal_prev_flow *= neewt_image if forward else tween_image
                temporal_next_flow *= tween_image
                temporal_prev_flow *= ckey['temporal_flow_factor']
                temporal_next_flow *= ckey['temporal_flow_factor'] if forward else rckey['temporal_flow_factor']

                # store up the remaps of the flow for each frame
                temporal_prev_flow_store = cv2.add(temporal_prev_flow_store, temporal_prev_flow)
                if idx > start_range:
                    temporal_next_flow_store = cv2.add(temporal_next_flow_store, temporal_next_flow)

                # do this frame's transform which includes all past flows from this cadence cycle
                prev = image_transform_optical_flow(prev, temporal_prev_flow_store)
                if idx > start_range:
                    next = image_transform_optical_flow(next, temporal_next_flow_store)

                # downscale and store frames
                if updown_scale > 1:
                    prev, _ = updown_scale_whatever(prev, shape=prev_shape)
                    next, _ = updown_scale_whatever(next, shape=next_shape)

                # update turbo dict
                turbo[idx]['prev'] = np.copy(prev)
                turbo[idx if forward else rdx]['next'] = np.copy(next)
        pbar.close()

    return turbo

