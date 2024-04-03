import numpy as np
from tqdm import tqdm
from .cadence import set_range_get_total, better_range, reverse_index
from .hybrid_render import hybrid_motion, image_transform_optical_flow
from .hybrid_ransac import image_transform_ransac
from .hybrid_render import get_default_motion

# modifies turbo object for hybrid motion
def cadence_hybrid_motion(turbo, inputfiles, inputframes_path, hybridframes_path, args, anim_args, raft_model):
    if anim_args.hybrid_motion != 'None':
        start_range = turbo['idx_start']
        end_range = turbo['idx_end']
        dimensions = (turbo[start_range]['prev'].shape[1], turbo[start_range]['prev'].shape[0]) 
        prev_motion = get_default_motion(anim_args.hybrid_motion, dimensions)
        next_motion = get_default_motion(anim_args.hybrid_motion, dimensions)
        motions = []
        last_prev_img = None
        last_next_img = None
        matrix_flow = None
        turbo_steps = anim_args.diffusion_cadence

        # report to console
        hybrid_motion_type_msg = f"{anim_args.hybrid_flow_method if anim_args.hybrid_motion in ['Optical Flow', 'Matrix Flow'] else ''} {anim_args.hybrid_motion}"
        print(f"Cadence calculating & applying {hybrid_motion_type_msg} hybrid motion to frames {start_range}-{end_range-1}")

        hybrid_motion_args = (args, anim_args, inputfiles, hybridframes_path)
        hybrid_motion_kwargs = {'image_return': True, 'suppress_console': True}

        # set up loop range and total
        r, rt = set_range_get_total(start_range, end_range-1, 1, False)
               
        # loop over turbo and make changes to prev, with progress bar
        pbar = tqdm(total=rt)
        for idx in better_range(r[0], r[1], r[2], r[3]):
            pbar.set_description(f"Cadence hybrid motion: {idx}/{r[0]}-{r[1]-1} ")
            pbar.update(1)

            # hybrid motion during cadence (outer loop already did this on last cadence frame pre-generation)
            rdx = reverse_index(idx, r[0], r[1])
            prev = turbo[idx]['prev']
            next = turbo[rdx]['next']

            if 'idx_start' in turbo['turbo_last']:
                # establish index for images one cadence cycle ago and retrieve images
                last_prev_img = np.copy(turbo['turbo_last'][idx-turbo_steps]['prev'])
                last_next_img = np.copy(turbo['turbo_last'][rdx-turbo_steps]['next'])
            elif turbo['last_img'] is not None:
                # if first cadence cycle, it must use the last_img (which is the frame 0 image)
                last_prev_img = np.copy(turbo['last_img'])
                last_next_img = np.copy(turbo['last_img'])

            if anim_args.hybrid_motion == 'Matrix Flow':
                matrix_flow = turbo['matrix_flow']
                hybrid_motion_kwargs.update({'matrix_flow': matrix_flow})

            # prev section - catch up with current state by warping with the flow list so far
            for prev_motion, next_motion in motions:
                if anim_args.hybrid_motion in ['Perspective', 'Affine']:
                    prev = image_transform_ransac(prev, prev_motion, anim_args.hybrid_motion)
                    next = image_transform_ransac(next, next_motion, anim_args.hybrid_motion)
                else:
                    prev = image_transform_optical_flow(prev, prev_motion, 1)
                    next = image_transform_optical_flow(next, next_motion, 1)
            
            # process hybrid motion on prev frame
            prev, prev_motion = hybrid_motion(idx, prev, last_prev_img, prev_motion, *hybrid_motion_args, float(turbo[idx]['ckey']['hybrid_flow_factor']), raft_model, **hybrid_motion_kwargs)
            next, next_motion = hybrid_motion(rdx, next, last_next_img, next_motion, *hybrid_motion_args, float(turbo[rdx]['ckey']['hybrid_flow_factor']), raft_model, **hybrid_motion_kwargs, reverse=True)
            motions.append((prev_motion, next_motion))
                
            turbo[idx]['prev'] = np.copy(prev)
            turbo[rdx]['next'] = np.copy(next)

            last_prev_img = np.copy(prev)
            last_next_img = np.copy(next)

        pbar.close()

    return turbo
