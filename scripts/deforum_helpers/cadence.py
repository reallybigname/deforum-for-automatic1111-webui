import cv2
import numpy as np
import os
import math
from typing import Generator, Optional
from tqdm import tqdm
from .easing import easing_linear_interp
from .animation import anim_frame_warp, depth_normalize

# creates turbo dict which other functions modify the images in
def build_turbo_dict(turbo_last, img1, img2, idx_start, idx_end, keys, last_img, last_hybrid_motion):
    ''' Cadence notes:
        - the turbo next image is not animated on the first frame (already moved in outer loop)
    '''
    # cadence keys needed for entire cadence cycle, and tweens derived from ckeys
    ckeys = get_cadence_keys_range(keys, idx_start, idx_end)
    
    # turbo dict creation
    turbo = {
        'idx_start': idx_start,
        'idx_end': idx_end,
        'last_img': last_img,
        'last_hybrid_motion': last_hybrid_motion,
        'prev': img1,
        'next': img2,
        'turbo_last': turbo_last
    }

    # insert prev and next in each index
    for idx in better_range(idx_start, idx_end):
        turbo[idx] = {
            'prev': img1,
            'next': img2,
            'ckey': ckeys[idx]
        }

    return turbo

def cadence_animation_frames(turbo, depth, depth_model, inputfiles, hybridframes_path, anim_args, keys, root, z_translation_list=None):
    ''' Cadence notes:
        - the turbo next image is not animated on the first frame (already moved in outer loop)
    '''   
    idx_start, idx_end = turbo['idx_start'], turbo['idx_end']
    prev, next = turbo['prev'], turbo['next']
    translation_z_estimates = []

    # report to console that cadence is animating
    print(f"\033[94mDiffusion cadence:\033[0m Animating {anim_args.diffusion_cadence} frames from {idx_start} to {idx_end-1}")

    # builds dict for cadence, fully animated but not mixed
    r, rt = set_range_get_total(idx_start, idx_end, 1, False)
    pbar = tqdm(total=rt)
    for idx in better_range(r[0], r[1], r[2], r[3]):
        pbar.set_description(f"Cadence animating frames {r[0]}→{r[1]-1} ")
        pbar.update(1)

        # need tween for tween['image']
        tween = turbo[idx]['ckey']['tween']

        # if none of these conditions are met, previous depth gets passed through anyway
        if depth_model is not None:
            # IN PROGRESS - blend_depth is experimental. set to True to try it. it works.
            blend_prev_depth = False

            depth_next = depth_model.predict(next, anim_args.midas_weight, root.half_precision)

            if blend_prev_depth:
                # experimental - blend prev and next tensors using tween for images as alpha/beta of blend
                depth_prev = depth_model.predict(prev, anim_args.midas_weight, root.half_precision)
                depth = depth_next * tween['image'] + depth_prev * (1 - tween['image'])
            else:
                # original behavior, just gets depth from turbo next image
                depth = depth_next

        # same args for animation of prev/next
        anim_frame_warp_args = (anim_args, keys, idx, depth_model, depth, root.device, root.half_precision)
        anim_frame_warp_kwargs = {'inputfiles': inputfiles, 'hybridframes_path': hybridframes_path, 'z_translation_list': z_translation_list}

        # animate prev image
        prev, _, prev_matrix_flow, translation_z_estimate = anim_frame_warp(prev, *anim_frame_warp_args, **anim_frame_warp_kwargs)

        # animate next image, unless first frame. skip 1st frame since it was already animated pre-generation
        if idx > idx_start:
            next, _, next_matrix_flow, translation_z_estimate = anim_frame_warp(next, *anim_frame_warp_args, **anim_frame_warp_kwargs)

        # structure of turbo dict index
        turbo[idx].update({
            "prev": prev,   # bgr nd array
            "next": next,   # bgr nd array
            "depth": depth  # depth tensor
        })
        
        # matrix flow stores itself in turbo index as prev and next tuples (matrix, flow)
        if anim_args.hybrid_motion != 'None':
            turbo[idx].update({
                "matrix_flow": next_matrix_flow if idx > idx_start else prev_matrix_flow
            })

    translation_z_estimates.append(translation_z_estimate)

    pbar.close()

    return turbo, translation_z_estimates

def get_cadence_keys_range(keys, idx_start, idx_end):
    response = {
        'idx_start': idx_start,
        'idx_end': idx_end
    }
    for idx in range(idx_start, idx_end+1):
        tween_prime = get_tween(idx, idx_start, idx_end)
        response[idx] = get_cadence_keys(keys, idx, tween_prime)
    return response

# idx_end is one higher then end, just like with range()
def get_tween(current_idx, idx_start, idx_end):
    return float(current_idx - idx_start + 1) / float(idx_end - idx_start)

def get_cadence_keys(keys, idx, tween_prime):
    # gets all the pertinent keys and tween values that need to be available during cadence
    cadence_diffusion_easing = float(keys.cadence_diffusion_easing_schedule_series[idx])
    cadence_flow_easing = float(keys.cadence_flow_easing_schedule_series[idx])
    return {
        'cadence_diffusion_easing': cadence_diffusion_easing,
        'cadence_flow_easing': cadence_flow_easing,
        'cadence_flow_factor': float(keys.cadence_flow_factor_schedule_series[idx]),
        'cadence_flow_warp_factor': str(keys.cadence_flow_warp_factor_schedule_series[idx]),
        'hybrid_comp_conform': float(keys.hybrid_comp_conform_schedule_series[idx]),
        'hybrid_flow_factor': float(keys.hybrid_flow_factor_schedule_series[idx]),
        'temporal_flow_factor': float(keys.temporal_flow_factor_schedule_series[idx]),
        'temporal_flow_motion_stabilizer_factor': float(keys.temporal_flow_motion_stabilizer_factor_schedule_series[idx]),
        'temporal_flow_rotation_stabilizer_factor': float(keys.temporal_flow_rotation_stabilizer_factor_schedule_series[idx]),
        'temporal_flow_target_frame': math.floor(keys.temporal_flow_target_frame_schedule_series[idx]),
        'morpho_flow_factor': float(keys.morpho_flow_factor_schedule_series[idx]),
        'morpho': str(keys.morpho_schedule_series[idx]),
        'morpho_iterations': int(keys.morpho_iterations_schedule_series[idx] // 1),
        'cc_alpha': float(keys.color_coherence_alpha_schedule_series[idx]),
        'loop_comp_type': str(keys.loop_comp_type_schedule_series[idx]),
        'loop_comp_alpha': float(keys.loop_comp_alpha_schedule_series[idx]),
        'loop_comp_conform': float(keys.loop_comp_conform_schedule_series[idx]),
        'tween': {
            'prime': tween_prime,
            "image": easing_linear_interp(tween_prime, cadence_diffusion_easing),
            "flow": easing_linear_interp(tween_prime, cadence_flow_easing)
        }
    }

def blend_depth(depth1, alpha, depth2, beta):
    return (depth1 * alpha + depth2 * beta) / (alpha + beta)

def cadence_save(string, index, outdir, opencv_image, save_depth_maps, depth_model, depth, midas_weight, cmd_opts, lowvram, sd_hijack, sd_model, devices, root):
    # generate filename with string and index and write image to file
    filename = f"{string}_{index:09}.png"
    cv2.imwrite(os.path.join(outdir, filename), opencv_image)

    # handle saving of depth maps    
    if save_depth_maps and depth_model is not None:
        if cmd_opts.lowvram or cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
            sd_hijack.model_hijack.undo_hijack(sd_model)
            devices.torch_gc()
            depth_model.to(root.device)

        # distinguish between depth already captured and depth needing to be captured
        if depth is None:
            print("=================DEPTH: Depth NOT provided - predicted by save function")
            # if depth is not provided, we can predict depth and just use the model's save function
            depth = depth_model.predict(opencv_image, midas_weight, root.half_precision)
            depth_model.save(os.path.join(outdir, f"{string}_depth_{index:09}.png"), depth, output_format="opencv")
        else:
            print("=================DEPTH: Depth provided - normalized and converted to image by save function")
            # if depth already captured, we have to normalize depth 0-1, which is required for visualizing from 0-255
            # (our depth is from 1 to 2 in order to make animation work right)
            depth = depth_normalize(depth)
            depth_image = depth_model.to_image(depth, output_format='opencv')
            cv2.imwrite(os.path.join(outdir, f"{string}_depth_{index:09}.png"), depth_image)

        if cmd_opts.lowvram or cmd_opts.medvram:
            depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)

def print_cadence_msg(tween_frame_idx, optical_flow_cadence, ckey, tween, max_frames):
    # print cadence creation message when it starts a cadence cycle
    list_tweens = []
    msg_optflow = None
    msg = f"\033[94mCadence frame complete:\033[0m {tween_frame_idx} of [0→{max_frames}] | "

    if ckey['cadence_diffusion_easing'] != 0:
        list_tweens.append(f"image:{tween['image']:0.2f}")
    if optical_flow_cadence != 'None':
        if ckey['cadence_flow_easing'] != 0:
            list_tweens.append(f"flow:{tween['flow']:0.2f}")
        msg_optflow = f"flow:{optical_flow_cadence} x {ckey['cadence_flow_factor']:0.2f}"
    
    if len(list_tweens) == 0:
        msg += f"tween:{tween['prime']:0.2f}"
    else:
        msg += f"tweens[prime:{tween['prime']:0.2f} {' '.join(list_tweens)}]"
    if msg_optflow is not None:
        msg += f" | {msg_optflow}"

    return print(msg)

# retrieves commonly needed vars for cadence
def get_cadence_loop_vars(turbo, idx, ckeys, tweens, prev_idx=None, next_idx=None):
    ckey = ckeys[idx]
    tween = tweens[idx]
    idx_p = idx if prev_idx is None else prev_idx
    idx_n = idx if next_idx is None else next_idx 
    this_prev = turbo[idx_p]['prev']
    this_next = turbo[idx_n]['next']
    prev = None if this_prev is None else np.copy(this_prev)
    next = None if this_next is None else np.copy(this_next)
    return prev, next, ckey, tween

# establish range for loop, get total for progress bar: floor((end_range - start_range) / increment)
def set_range_get_total(start_range, end_range, increment, reverse=False):
    range = [start_range, end_range, increment, reverse]
    floor_total = ((range[1] - range[0]) / range[2]) // 1
    return range, floor_total

# replicates behavior of range but only if a start is provided (makes start into end, starts at 0)
# has reverse flag and can reverse without changing range vars 
# has inclusive flag for making a loop include the end range, rather than it being the stopper at +1 
def better_range(start: int, end: Optional[int] = None, step: int = 1, reverse: bool = False, inclusive: bool = False) -> Generator[int, None, None]:
    if inclusive:
        if end is None:
            raise ValueError("End value must be defined if inclusive mode is used.")
        else:
            end += 1
    else:
        if end is None:
            end = start
            start = 0
    if reverse:
        start, end = end - 1, start - 1
        step = -step
    yield from range(start, end, step)

def reverse_index(index, start, end):
    vshift = 1 - start
    vend = end + vshift
    vindex = index + vshift
    vrindex = vend - vindex

    return vrindex - vshift