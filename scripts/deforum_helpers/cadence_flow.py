import cv2
import re
from tqdm import tqdm
from .hybrid_flow import image_transform_optical_flow, get_flow_from_images, remap_flow, combine_flow_fields
from .cadence import set_range_get_total, better_range

# modifies turbo object for optical flow cadence
def cadence_flow(turbo, flow_method, raft_model, updown_scale=1):
    ''' bi-directional flow with easing setup
        - accepts a turbo dict with all the indexed cadence animation frames for prev and next separately
        - flow increments flow_from_start and flow_end are smoothly blended using the same tween value for flow as the blend's alpha/beta mix
        - tween/neewt range is 0 to 1. A neewt is an inverse tween (1 - tween) 
    '''
    # cadence flow
    if flow_method != 'None':       
        # establish range from turbo dict
        start_range = turbo['idx_start']
        end_range = turbo['idx_end']
        frame_count = end_range - start_range

        # establish total and range for loop, end_range-1 to preserve last frame
        r, rt = set_range_get_total(start_range, end_range-1, 1)

        # shared flow arguments
        flow_args = (flow_method, raft_model, updown_scale)
        flow_kwargs = {'anchoring': False}
        flows = {}

        # MOVEMENT flow setup - get main images that cadence cycle was built from as a baseline
        turbo_prev_image = turbo['prev']
        turbo_next_image = turbo['next']
        
        # SHAPE flow setup - get flows from (prev_first to next_first) for prev | (prev_last to next_last) for next
        shape_flow_prev = get_flow_from_images(turbo[start_range]['prev'], turbo[start_range]['next'], *flow_args, **flow_kwargs)
        shape_flow_next = get_flow_from_images(turbo[end_range-1]['prev'], turbo[end_range-1]['next'], *flow_args, **flow_kwargs)
        
        # MOVEMENT flow section - collect all flows from turbo_prev_image to all prevs and turbo_next_image to all nexts in temporary flows dict
        pbar = tqdm(total=rt)
        for idx in better_range(r[0], r[1], r[2], r[3]):
            pbar.set_description(f"Cadence flow getting baseline to present flows: ({r[0]}→{r[1]-1}) ")
            pbar.update(1)

            ff = turbo[idx]['ckey']['cadence_flow_factor']
            if ff != 0:
                prev_item = get_flow_from_images(turbo_prev_image, turbo[idx]['prev'], *flow_args, **flow_kwargs)
                next_item = get_flow_from_images(turbo_next_image, turbo[idx]['next'], *flow_args, **flow_kwargs)
                flows[idx] = [prev_item, next_item]
        pbar.close()

        # SHAPE flow section
        pbar2 = tqdm(total=rt)
        for idx in better_range(r[0], r[1], r[2], r[3]):
            pbar2.set_description(f"Cadence flow processing frames: ({r[0]}→{r[1]-1}) ")
            pbar2.update(1)

            # warps next image back by warp factor towards the prev state while advancing and tweening prev and next
            ff = turbo[idx]['ckey']['cadence_flow_factor']
            if ff != 0:
                shapes_prev, shapes_next, weights = [], [], []
                tween_flow = turbo[idx]['ckey']['tween']['flow']
                
                # get factors and weights from key, split into a list of factor/weight pairs, then create array from those to loop over
                warp_factor_list_raw = turbo[idx]['ckey']['cadence_flow_warp_factor']
                warp_factor_list = extract_bracket_content(warp_factor_list_raw)
                warp_factors_weights = []
                for fw in warp_factor_list:
                    wfw = fw.split('|')
                    factor = float(wfw[0])
                    weight = 1.0 if len(wfw) == 1 else float(wfw[1])
                    warp_factors_weights.append([factor, weight])

                # loop over warp factors and weights, build shapes prev/next lists and weight list
                for factor, weight in warp_factors_weights:
                    # increments are achieved through the warp factor
                    s_prev_factored = shape_flow_prev * factor
                    s_next_factored = shape_flow_next * factor

                    # advance prev and next with tween, next's flow has the warp proportion subtracted
                    shapes_prev.append(s_prev_factored * tween_flow)
                    shapes_next.append(cv2.subtract(s_next_factored * tween_flow, s_next_factored))
                    weights.append(weight)
                    
                # SHAPE flows from weighted blend of flows for prev and next with weighting
                shape_prev = combine_flow_fields(weights, shapes_prev)
                shape_next = combine_flow_fields(weights, shapes_next)

                # SHAPE & MOVEMENT combined by remapping shape flow using movement flow
                shape_and_move_prev = remap_flow(shape_prev, flows[idx][0]) # * ff
                shape_and_move_next = remap_flow(shape_next, flows[idx][1]) # * ff

                img_prev = image_transform_optical_flow(turbo[idx]['prev'], shape_and_move_prev, ff)
                img_next = image_transform_optical_flow(turbo[idx]['next'], shape_and_move_next, ff)

                turbo[idx]['prev'] = img_prev
                turbo[idx]['next'] = img_next

        pbar2.close()
        
    return turbo

def extract_bracket_content(s):
    return re.findall(r'\[(.*?)\]', s)
