import cv2
import os
import numpy as np
from tqdm import tqdm
from .cadence import set_range_get_total, better_range, reverse_index
from .hybrid_flow import image_transform_optical_flow
from .conform import conform_images
from .video_audio_utilities import get_frame_name
from .load_images import load_image

# modifies turbo object for hybrid compositing conformation
def cadence_hybrid_comp(turbo, inputframes_path, dimensions, args, anim_args, raft_model):
    ''' this function doesn't composite, but allows for the optional conform settings to work during cadence '''
    start_range = turbo['idx_start']
    end_range =   turbo['idx_end']
    video_images = {}
    prev_flows = []
    next_flows = []
    conform_kwargs = {
        'flow_method': anim_args.hybrid_comp_conform_method,
        'raft_model': raft_model,
        'return_flow': True,
        'iterations': anim_args.hybrid_comp_conform_iterations
    }

    # report to console
    print(f"Cadence hybrid compositing conforming frame shape based on video shape using {anim_args.hybrid_comp_conform_method} flow on frames {start_range}-{end_range-1}")

    # prev loop, set up range and total, loop over videos and get frames, conform prev in turbo object to video
    r, rt = set_range_get_total(start_range, end_range, 1, reverse=False)
    pbar = tqdm(total=rt)
    for idx in better_range(r[0], r[1], r[2], r[3]):
        pbar.set_description(f"Cadence hybrid composite conform prev step: {idx}/{r[0]}-{r[1]-1} ")
        pbar.update(1)       
        ckey = turbo[idx]['ckey']
        # alpha = ckey['tween']['image']
        prev = np.copy(turbo[idx]['prev'])
        frame_path = os.path.join(inputframes_path, get_frame_name(anim_args.video_init_path) + f"{idx:09}.jpg")
        video_images[idx] = load_image(args.init_image, return_np_bgr=True) if anim_args.hybrid_use_init_image else cv2.imread(frame_path)
        video_images[idx] = cv2.resize(video_images[idx], dimensions, interpolation=cv2.INTER_LANCZOS4)    
                       
        # redo all flows thus far in loop
        for flow in prev_flows:
            prev = image_transform_optical_flow(prev, flow, 1)

        # conform prev to video index and write prev to turbo
        prev_dict = conform_images(prev, video_images[idx], alpha=ckey['hybrid_comp_conform'], **conform_kwargs)
        turbo[idx]['prev'] = prev_dict['image1']
        
        # save list of flows from conformations to repeat successively in catch-up loop above
        prev_flows.append(prev_dict['flow1'])
    pbar.close()

    # next loop, set up reverse range and total, loop over turbo and make changes to next, with progress bar
    r, rt = set_range_get_total(start_range, end_range, 1, reverse=True)
    pbar = tqdm(total=rt)
    for idx in better_range(r[0], r[1], r[2], r[3]):
        pbar.set_description(f"Cadence hybrid composite conform next step: {idx}/{r[0]}-{r[1]-1} ")
        pbar.update(1)
        ckey = turbo[idx]['ckey']
        # prev = np.copy(turbo[idx]['prev'])
        next = np.copy(turbo[idx]['next'])

        # redo all flows thus far in loop
        for flow in next_flows:
            next = image_transform_optical_flow(next, flow, 1)

        # conform next to video index
        next_dict = conform_images(next, video_images[idx], alpha=ckey['hybrid_comp_conform'], **conform_kwargs)
        next_flows.append(next_dict['flow1'])

        # # now that we have both, conform prev and next together in a final alignment
        # conform_dict = conform_images(prev, next_dict['image1'], alpha=0.5, **conform_kwargs)

        # write the conformed image back to turbo
        # turbo[idx]['prev'] = np.copy(conform_dict['image1'])
        turbo[idx]['next'] = np.copy(next_dict['image1'])
    pbar.close()

    return turbo
