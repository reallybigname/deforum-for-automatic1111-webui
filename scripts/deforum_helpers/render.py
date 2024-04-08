import os
import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import time
import math

from PIL import Image
from .generate import generate, isJson
from .noise import add_noise
from .animation_key_frames import DeforumAnimKeys, LooperAnimKeys
from .video_audio_utilities import get_frame_name, get_next_frame, extract_video_init_without_hybrid, extract_video_for_cc_path
from .depth import DepthModel
from .parseq_adapter import ParseqAnimKeys
from .seed import next_seed
from .image_sharpening import unsharp_mask
from .load_images import get_mask, load_img, get_mask_from_file
from .composable_masks import compose_mask_with_check
from .settings import save_settings_from_animation_run
from .deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from .subtitle_handler import init_srt_file, write_frame_subtitle, format_animation_params
from .prompt import prepare_prompt

# reallybigname - dev note: I put this block together for easy reloading stuff I work on a lot with importlib
from .animation import anim_frame_warp
from .cadence import build_turbo_dict, cadence_animation_frames, print_cadence_msg, cadence_save
from .cadence_morpho import cadence_morpho
from .cadence_temporal_flow import cadence_temporal_flow
from .cadence_hybrid_comp import cadence_hybrid_comp
from .cadence_hybrid_motion import cadence_hybrid_motion
from .cadence_flow import cadence_flow
from .colors import color_coherence, get_cc_sample_from_video_frame, get_cc_sample_from_image_path
from .composite import make_composite_with_conform
from .displayer import IntervalPreviewDisplayer
from .generation import optical_flow_generation, redo_generation
from .history import ThingHistory
from .hybrid_video import hybrid_generation
from .hybrid_compositing import hybrid_composite
from .hybrid_render import hybrid_motion, get_default_motion
from .image_functions import bgr2gray_bgr, bgr_np2pil_rgb
from .masks import do_overlay_mask
from .morphological import image_morphological_flow_transform
from .resume import get_resume_vars
from .temporal_flow import do_temporal_flow
from .updown_scale import updown_scale_to_integer

from modules.shared import opts, cmd_opts, state, sd_model
from modules import lowvram, devices, sd_hijack
from .RAFT import RAFT

# IN PROGRESS
# START reallybigname dev code to reload stuff (to be removed)
import sys
import importlib
importlib.reload(sys.modules['deforum_helpers.animation'])
importlib.reload(sys.modules['deforum_helpers.cadence'])
importlib.reload(sys.modules['deforum_helpers.cadence_morpho'])
importlib.reload(sys.modules['deforum_helpers.cadence_temporal_flow'])
importlib.reload(sys.modules['deforum_helpers.cadence_hybrid_comp'])
importlib.reload(sys.modules['deforum_helpers.cadence_hybrid_motion'])
importlib.reload(sys.modules['deforum_helpers.cadence_flow'])
importlib.reload(sys.modules['deforum_helpers.colors'])
importlib.reload(sys.modules['deforum_helpers.composite'])
importlib.reload(sys.modules['deforum_helpers.displayer'])
importlib.reload(sys.modules['deforum_helpers.generation'])
importlib.reload(sys.modules['deforum_helpers.history'])
importlib.reload(sys.modules['deforum_helpers.hybrid_video'])
importlib.reload(sys.modules['deforum_helpers.hybrid_compositing'])
importlib.reload(sys.modules['deforum_helpers.hybrid_render'])
importlib.reload(sys.modules['deforum_helpers.image_functions'])
importlib.reload(sys.modules['deforum_helpers.masks'])
importlib.reload(sys.modules['deforum_helpers.morphological'])
importlib.reload(sys.modules['deforum_helpers.resume'])
importlib.reload(sys.modules['deforum_helpers.temporal_flow'])
importlib.reload(sys.modules['deforum_helpers.updown_scale'])
from .animation import anim_frame_warp
from .cadence import build_turbo_dict, cadence_animation_frames, print_cadence_msg, cadence_save
from .cadence_morpho import cadence_morpho
from .cadence_temporal_flow import cadence_temporal_flow
from .cadence_hybrid_comp import cadence_hybrid_comp
from .cadence_hybrid_motion import cadence_hybrid_motion
from .cadence_flow import cadence_flow
from .colors import color_coherence, get_cc_sample_from_video_frame, get_cc_sample_from_image_path
from .composite import make_composite_with_conform
from .displayer import IntervalPreviewDisplayer
from .generation import optical_flow_generation, redo_generation
from .history import ThingHistory
from .hybrid_video import hybrid_generation
from .hybrid_compositing import hybrid_composite
from .hybrid_render import hybrid_motion, get_default_motion
from .image_functions import bgr2gray_bgr, bgr_np2pil_rgb
from .masks import do_overlay_mask
from .morphological import image_morphological_flow_transform
from .resume import get_resume_vars
from .temporal_flow import do_temporal_flow
from .updown_scale import updown_scale_to_integer
# END
    
def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):

    #_/‾𝓐𝓷𝓲𝓶𝓪𝓽𝓲𝓸𝓷‾𝓢𝓮𝓽𝓾𝓹‾‾‾‾‾\_____________
       
    # read auto1111's live preview setting for restoration when the function completes loop (we will change this var back and forth during)
    # set up smooth timed preview display mechanism - starts on instantiation here
    live_previews_enable_store = opts.live_previews_enable
    disp = IntervalPreviewDisplayer(frames_per_cycle=anim_args.diffusion_cadence)

    # srt files
    if opts.data.get("deforum_save_gen_info_as_srt", False):  # create .srt file and set timeframe mechanism using FPS
        srt_filename = os.path.join(args.outdir, f"{root.timestring}.srt")
        srt_frame_duration = init_srt_file(srt_filename, video_args.fps)

    # vars used often, shortened for clarity
    mode = anim_args.animation_mode
    dimensions = (args.W, args.H)
    updown_scale = updown_scale_to_integer(anim_args.updown_scale)
    inputframes_path = os.path.join(args.outdir, 'inputframes')
    maskframes_path = os.path.join(args.outdir, 'maskframes')
    hybrid_cn_paths = []
    inputfiles = None
    cn_files = None
    hybridframes_path = None

    # handle controlnet video input frames generation (before hybrid, so it can use the paths)
    if is_controlnet_enabled(controlnet_args):
        hybrid_cn_paths = unpack_controlnet_vids(args, anim_args, controlnet_args)

    # specific to 2D/3D modes
    if mode in ['2D', '3D']:
        ''' handle hybrid video setup and frame generation '''
        # hybrid overlay var needs to be defined for hybrid compositing of any kind 
        if anim_args.hybrid_composite != 'None':
            hybrid_overlay = None
        # if hybrid compositing is being used, or if motion is being used, generate inputframes and hybrid frame path
        if anim_args.hybrid_composite != 'None' or anim_args.hybrid_motion != 'None':
            args, anim_args, inputfiles, cn_files, hybridframes_path = hybrid_generation(args, anim_args, inputframes_path, hybrid_cn_paths, root)
            # for color coherence using video
            cc_vid_folder, cc_vid_path = 'inputframes', anim_args.video_init_path
        # auto-extract in the case that cc source is 'Video Init' but no hybrid features to extract video
        elif anim_args.color_coherence != 'None' and anim_args.color_coherence_source == 'Video Init':
            cc_vid_folder, cc_vid_path = extract_video_init_without_hybrid(args.outdir, anim_args, inputframes_path)
        
        # for debugging color coherence: set to True to save the pixel-mixed image & sample frames to see if the cc alpha is working properly 
        save_cc_mix = False
        if save_cc_mix:
            cc_mix_outdir = os.path.join(args.outdir, "ccmixed")
            print(f"Saving color coherence mixed sample frames to:\n{cc_mix_outdir}")
            if not os.path.exists(cc_mix_outdir):
                os.makedirs(cc_mix_outdir)
        else:
            cc_mix_outdir = None

        # auto-extract in the case that cc source is custom 'Video Path' (could also have a video init)
        if anim_args.color_coherence_source == 'Video Path':
            cc_vid_folder, cc_vid_path = extract_video_for_cc_path(args.outdir, anim_args, 'ccvideoframes')

        # guided images looper setup
        if loop_args.use_looper:
            print("Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
            if args.strength == 0:
                raise RuntimeError("Strength needs to be greater than 0 in Init tab")
            args.strength_0_no_init = False
            args.seed_behavior = "schedule"
            if not isJson(loop_args.init_images):
                raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

    # set up color matcher if using color coherence
    if anim_args.color_coherence != 'None':
        color_matcher = color_coherence()

    # use parseq if manifest is provided
    use_parseq = parseq_args.parseq_manifest is not None and parseq_args.parseq_manifest.strip()

    # expand key frame strings to values
    keys = DeforumAnimKeys(anim_args, args.seed) if not use_parseq else ParseqAnimKeys(parseq_args, anim_args, video_args)
    loopSchedulesAndData = LooperAnimKeys(loop_args, anim_args, args.seed)

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to:\n{args.outdir}")
    
    # create output folder for turbo frames if needed
    if anim_args.cadence_save_turbo_frames:
        outdir_turbo = os.path.join(args.outdir, "turbo")
        print(f"Saving turbo frames to:\n{outdir_turbo}")
        if not os.path.exists(outdir_turbo):
            os.makedirs(outdir_turbo)
        
    # save settings.txt file for the current run
    save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    # state for interpolating between diffusion steps
    turbo_steps = 1 if mode == 'Video Input' else int(anim_args.diffusion_cadence)

    # set master frame index, initialize crucial vars
    frame_idx = 0
    cc_sample = None
    prev_img = None
    prev_motion = get_default_motion(anim_args.hybrid_motion, dimensions)
    turbo_prev_image = None
    turbo_next_image = None
    turbo = {}
    matrix_flow = None
    
    # IN PROGRESS special debug var for visualizing the conform function used in hybrid compositing, but only is save extra frames is checked and this var is true
    return_conform_iterations_debug = True
    return_conform_iterations = anim_args.hybrid_comp_save_extra_frames and return_conform_iterations_debug

    # determine how many img history states are needed
    if anim_args.temporal_flow == "None":
        img_history_max_states = 1 + turbo_steps
    else:
        img_history_max_states = 1 + (max(keys.temporal_flow_target_frame_schedule_series) + turbo_steps)

    # set up histories (auto-deleting as more are added up to max_states)
    img_history = ThingHistory(max_states=img_history_max_states)       # every frame is stored here as it is saved, whether in main loop at cadence=1 or every frame from cadence loops
    turbo_history = ThingHistory(max_states=1)                          # the turbo cadence object history (always refers to last turbo from history)
    hybrid_motion_history = ThingHistory(max_states=turbo_steps + 1)    # hybrid flow history allows cadence to start with the proper hybrid flow history from the last cadence end
    z_translation_history = ThingHistory(max_states=25)                 # a history of z_translations, used in animation.py for matrix_flow

    # resume from timestring - get vars to resume from files (requires at least 3 "actual" frames - see function)
    if anim_args.resume_from_timestring:
        root.timestring = anim_args.resume_timestring # set timestring in root for resume
        frame_idx, prev_img, img_history = get_resume_vars(args.outdir, anim_args.resume_timestring, turbo_steps, img_history)

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True

    # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()

    # load depth model for 3D if required for animation or for hybrid compositing
    predict_depths = (mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    predict_depths_for_hybrid = anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth']
    if predict_depths or predict_depths_for_hybrid:
        keep_in_vram = opts.data.get("deforum_keep_3d_models_in_vram")
        device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else root.device)
        depth_model = DepthModel(root.models_path, device, root.half_precision, keep_in_vram=keep_in_vram, depth_algorithm=anim_args.depth_algorithm, Width=args.W, Height=args.H,
                                 midas_weight=anim_args.midas_weight)
    else:
        depth_model = None
        anim_args.save_depth_maps = False

    # load raft model if it is being used
    raft_model = None
    load_raft = (anim_args.optical_flow_cadence == "RAFT" and int(anim_args.diffusion_cadence) > 1) or \
                (anim_args.hybrid_flow_method == "RAFT" and anim_args.hybrid_motion == "Optical Flow") or \
                (anim_args.hybrid_composite != "None" and anim_args.hybrid_comp_conform_method != "None") or \
                (anim_args.loop_comp_conform_method == "RAFT" and all(x.lower() != 'none' for x in keys.loop_comp_type_schedule_series)) or \
                (anim_args.optical_flow_redo_generation == "RAFT") or \
                (anim_args.temporal_flow == "RAFT") or \
                (anim_args.morpho_flow == "RAFT")
    if load_raft:
        print("Loading RAFT model...")
        raft_model = RAFT()

    # reset the mask vals as they are overwritten in the compose_mask algorithm
    mask_vals = {}
    noise_mask_vals = {}
    mask_vals['everywhere'] = Image.new('1', dimensions, 1)
    noise_mask_vals['everywhere'] = Image.new('1', dimensions, 1)
    mask_image = None

    # get mask image if use_init true and init_image present (can't use mask without an init image)
    if args.use_init and args.init_image != None and args.init_image != '':
        _, mask_image = load_img(args.init_image, shape=dimensions, use_alpha_as_mask=args.use_alpha_as_mask)
        mask_vals['video_mask'] = mask_image
        noise_mask_vals['video_mask'] = mask_image

    # Grab the first frame masks since they wont be provided until next frame
    # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
    # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
    if anim_args.use_mask_video:
        args.mask_file = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        root.noise_mask = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        noise_mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
    elif mask_image is None and args.use_mask:
        mask_vals['video_mask'] = get_mask(args)
        noise_mask_vals['video_mask'] = get_mask(args)  # TODO?: add a different default noisc mask

    # acquire the sample just once for color coherence by 'Image Path' before loop
    if anim_args.color_coherence_source in ['Image Path']:
        cc_sample = get_cc_sample_from_image_path(anim_args.color_coherence_image_path, dimensions)

    # Webui
    state.job_count = anim_args.max_frames+1

    #_/‾𝓕𝓻𝓪𝓶𝓮‾𝓛𝓸𝓸𝓹‾𝓢𝓽𝓪𝓻𝓽‾‾‾‾‾\____________
    
    while frame_idx <= anim_args.max_frames:       
        # Webui - if using cadence, progress is tracked in turbo cadence loop
        if turbo_steps == 1:
            state.job = f"frame {frame_idx + 1}/{anim_args.max_frames+1}"
            state.job_no = frame_idx + 1
            if state.skipped:
                print("\n** PAUSED **")
                state.skipped = False
                while not state.skipped:
                    time.sleep(0.1)
                print("** RESUMING **")

        # get animation keys for this frame_idx
        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        scale = keys.cfg_scale_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        kernel = int(keys.kernel_schedule_series[frame_idx])
        sigma = keys.sigma_schedule_series[frame_idx]
        amount = keys.amount_schedule_series[frame_idx]
        threshold = keys.threshold_schedule_series[frame_idx]
        cc_alpha = keys.color_coherence_alpha_schedule_series[frame_idx] 
        loop_comp_type = keys.loop_comp_type_schedule_series[frame_idx]
        loop_comp_alpha = keys.loop_comp_alpha_schedule_series[frame_idx]
        loop_comp_conform = keys.loop_comp_conform_schedule_series[frame_idx]
        hybrid_comp_conform = keys.hybrid_comp_conform_schedule_series[frame_idx]
        hybrid_flow_factor = keys.hybrid_flow_factor_schedule_series[frame_idx]
        temporal_flow_factor = keys.temporal_flow_factor_schedule_series[frame_idx]
        temporal_flow_motion_stabilizer_factor = keys.temporal_flow_motion_stabilizer_factor_schedule_series[frame_idx]
        temporal_flow_rotation_stabilizer_factor = keys.temporal_flow_rotation_stabilizer_factor_schedule_series[frame_idx]
        temporal_flow_target_frame = math.floor(keys.temporal_flow_target_frame_schedule_series[frame_idx])
        morpho = keys.morpho_schedule_series[frame_idx]
        morpho_iterations = int(keys.morpho_iterations_schedule_series[frame_idx] // 1)
        morpho_flow_factor = keys.morpho_flow_factor_schedule_series[frame_idx]
        redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
        hybrid_comp_schedules = {
            "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_alpha": keys.hybrid_comp_mask_alpha_schedule_series[frame_idx],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_high": keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]
        }

        # vars start as None
        scheduled_sampler_name = None
        scheduled_clipskip = None
        scheduled_noise_multiplier = None
        scheduled_ddim_eta = None
        scheduled_ancestral_eta = None
        depth = None
        mask_seq = None
        noise_mask_seq = None

        # keys which may contain None or have other requirements
        if anim_args.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None:
            args.steps = int(keys.steps_schedule_series[frame_idx])
        if anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None:
            scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold()
        if anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None:
            scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx])
        if anim_args.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[frame_idx] is not None:
            scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[frame_idx])
        if anim_args.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[frame_idx] is not None:
            scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[frame_idx])
        if anim_args.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[frame_idx] is not None:
            scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[frame_idx])
        if args.use_mask and keys.mask_schedule_series[frame_idx] is not None:
            mask_seq = keys.mask_schedule_series[frame_idx]
        if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
            noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
        if args.use_mask and not anim_args.use_noise_mask:
            noise_mask_seq = mask_seq

        if mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
            # Unload the main checkpoint and load the depth model
            lowvram.send_everything_to_cpu()
            sd_hijack.model_hijack.undo_hijack(sd_model)
            devices.torch_gc()
            if predict_depths: depth_model.to(root.device)

        # get cc_sample for video on every frame by placing it before cadence & prev_img conditional block
        if anim_args.color_coherence != 'None' and anim_args.color_coherence_source in ['Video Init', 'Video Path'] and cc_alpha > 0:
            cc_sample = get_cc_sample_from_video_frame(anim_args.color_coherence_source, args.outdir, cc_vid_folder, get_frame_name(cc_vid_path), frame_idx, dimensions)

        # srt files outside cadence
        if turbo_steps == 1 and opts.data.get("deforum_save_gen_info_as_srt"):
            params_string = format_animation_params(keys, prompt_series, frame_idx)
            write_frame_subtitle(srt_filename, frame_idx, srt_frame_duration, f"F#: {frame_idx}; Cadence: false; Seed: {args.seed}; {params_string}")
            params_string = None

        # report that animation frame is in progress and what it's for
        if frame_idx <= anim_args.max_frames:
            if turbo_steps == 1 or frame_idx == 0:
                print(f"\033[36mFrame in progress:\033[0m {frame_idx} in [0→{anim_args.max_frames}] ")
            else:
                print(f"\033[36mFrame in progress:\033[0m {frame_idx} for cadence end range [{frame_idx+1-turbo_steps}→{frame_idx}] in [0→{anim_args.max_frames}] ")

        #_/‾𝓟𝓻𝓮𝓿_𝓲𝓶𝓰‾𝓢𝓮𝓬𝓽𝓲𝓸𝓷‾𝟏‾‾‾‾‾\__________

        if prev_img is not None:

            # __________Warping/Compositing__________

            # apply transforms to prev_img, get depth.
            if anim_args.animation_behavior == 'Normal':
                prev_img, depth, matrix_flow, translation_z_estimate = anim_frame_warp(np.copy(prev_img), anim_args, keys, frame_idx, depth_model=depth_model, depth=None,
                                                                                       device=root.device, half_precision=root.half_precision, inputfiles=inputfiles, hybridframes_path=hybridframes_path,
                                                                                       prev_flow=hybrid_motion_history.get_by_key_or_none(frame_idx-1),
                                                                                       prev_img=img_history.get_by_key_or_none(frame_idx-1),
                                                                                       raft_model=raft_model, z_translation_list=z_translation_history.list_states())
                z_translation_history.add_state(translation_z_estimate)

            # do hybrid compositing before motion
            if anim_args.hybrid_composite == 'Before Motion':
                prev_img, hybrid_overlay = hybrid_composite(args, anim_args, frame_idx, np.copy(prev_img), depth_model, hybrid_comp_schedules, inputframes_path, hybridframes_path, root,
                                                            conform=hybrid_comp_conform, raft_model=raft_model, return_iterations=return_conform_iterations)

            # hybrid video motion: Before Generation (normal) - if in cadence, it does a different cadence hybrid motion function
            if turbo_steps == 1 and anim_args.hybrid_motion_behavior == 'Before Generation' and anim_args.hybrid_motion != 'None':
                prev_img, prev_motion = hybrid_motion(frame_idx, np.copy(prev_img), img_history.get_by_key_or_none(frame_idx-1),
                                                      hybrid_motion_history.get_by_key_or_none(frame_idx-1), args, anim_args,
                                                      inputfiles, hybridframes_path, hybrid_flow_factor, raft_model, matrix_flow=matrix_flow)

            # do hybrid compositing after motion (normal)
            if anim_args.hybrid_composite == 'Normal':
                prev_img, hybrid_overlay = hybrid_composite(args, anim_args, frame_idx, np.copy(prev_img), depth_model, hybrid_comp_schedules, inputframes_path, hybridframes_path, root,
                                                            conform=hybrid_comp_conform, raft_model=raft_model, return_iterations=return_conform_iterations)

            # temporal flow = outside cadence behavior 'None' means it operates here only on the last turbo frame of each cycle
            temporal_history_length = temporal_flow_target_frame + turbo_steps
            if anim_args.temporal_flow != 'None' and temporal_flow_factor != 0 and img_history.length >= temporal_history_length:
                prev_img = do_temporal_flow(img_history[temporal_history_length], np.copy(prev_img), anim_args.temporal_flow, frame_idx, raft_model, flow_factor=temporal_flow_factor,
                                            motion_stabilizer_factor=temporal_flow_motion_stabilizer_factor, rotation_stabilizer_factor=temporal_flow_rotation_stabilizer_factor,
                                            return_target=anim_args.temporal_flow_return_target, target_frame=temporal_flow_target_frame, return_flow=False, updown_scale=updown_scale)

            # morphological transformation - outside cadence behavior 'None' means it operates here only on the last turbo frame of each cycle
            if anim_args.morpho_flow != 'None' and morpho_iterations != 0:
                prev_img = image_morphological_flow_transform(np.copy(prev_img), anim_args.morpho_image_type, anim_args.morpho_bitmap_threshold, anim_args.morpho_flow, morpho, morpho_iterations, frame_idx, raft_model, flow_factor=morpho_flow_factor, updown_scale=updown_scale, return_flow=False)

            # __________Colors/Filters__________

            # color coherence - before generation - make prev_img conform to cc_sample already collected (it should never be None at this point)
            if anim_args.color_coherence != 'None' and anim_args.color_coherence_behavior in ['Before', 'Before/After'] and cc_alpha > 0 and cc_sample is not None:
                prev_img = color_matcher.maintain_colors(np.copy(prev_img), cc_sample, cc_alpha, anim_args.color_coherence, frame_idx, console_msg="before generation", cc_mix_outdir=cc_mix_outdir, timestring=root.timestring)

            # force grayscale before generation - we force grayscale after the color match, before contrasting/blur/noise/etc
            if anim_args.color_force_grayscale == 'Before':
                prev_img = bgr2gray_bgr(np.copy(prev_img))

            # apply contrast
            prev_img = (np.copy(prev_img) * contrast).round().astype(np.uint8)

            # anti-blur
            if amount > 0:
                prev_img = unsharp_mask(np.copy(prev_img), (kernel, kernel), sigma, amount, threshold, mask_image if args.use_mask else None)

            # apply noise mask
            if args.use_mask or anim_args.use_noise_mask:
                root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq, noise_mask_vals, Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB)))

            # apply noise
            prev_img = add_noise(np.copy(prev_img), noise, args.seed, anim_args.noise_type, (anim_args.perlin_w, anim_args.perlin_h, anim_args.perlin_octaves, anim_args.perlin_persistence), root.noise_mask, args.invert_mask)
   
            #‾\_𝓔𝓷𝓭_𝓟𝓻𝓮𝓿_𝓲𝓶𝓰_𝓢𝓮𝓬𝓽𝓲𝓸𝓷_𝟏_____/‾‾‾‾‾‾

        # set args.scale based on schedule
        args.scale = scale

        # set strength based on schedule - Video Input/Interpolation use strength slider, not schedule
        if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
            args.strength = min(max(strength, 0.0), 1.0)

        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[frame_idx])

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]

        # seed scheduling
        if args.seed_behavior == 'schedule' or use_parseq:
            args.seed = int(keys.seed_schedule_series[frame_idx])

        # checkpoint scheduling
        if anim_args.enable_checkpoint_scheduling:
            args.checkpoint = keys.checkpoint_schedule_series[frame_idx]
        else:
            args.checkpoint = None

        # subseed scheduling
        if anim_args.enable_subseed_scheduling:
            root.subseed = int(keys.subseed_schedule_series[frame_idx])
            root.subseed_strength = float(keys.subseed_strength_schedule_series[frame_idx])

        # parsec
        if use_parseq:
            anim_args.enable_subseed_scheduling = True
            root.subseed = int(keys.subseed_schedule_series[frame_idx])
            root.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]

        # set value back into the prompt - prepare and report prompt and seed
        args.prompt = prepare_prompt(args.prompt, anim_args.max_frames, args.seed, frame_idx)

        # grab init image for current frame
        if mode == 'Video Input':
            args.init_image = get_next_frame(args.outdir, anim_args.video_init_path, frame_idx, False)
            print(f"Using video init frame { args.init_image}")

        # video mask
        if anim_args.use_mask_video:
            args.mask_file = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
            root.noise_mask = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
            mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

        # image mask
        if args.use_mask:
            prev_img_rgb = None if prev_img is None else Image.fromarray(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)) 
            args.mask_image = compose_mask_with_check(root, args, mask_seq, mask_vals, prev_img_rgb) if prev_img_rgb is not None else None  # we need it only after the first frame anyway

        # setting up some arguments for the looper
        loop_args.imageStrength = loopSchedulesAndData.image_strength_schedule_series[frame_idx]
        loop_args.blendFactorMax = loopSchedulesAndData.blendFactorMax_series[frame_idx]
        loop_args.blendFactorSlope = loopSchedulesAndData.blendFactorSlope_series[frame_idx]
        loop_args.tweeningFrameSchedule = loopSchedulesAndData.tweening_frames_schedule_series[frame_idx]
        loop_args.colorCorrectionFactor = loopSchedulesAndData.color_correction_factor_series[frame_idx]
        loop_args.use_looper = loopSchedulesAndData.use_looper
        loop_args.imagesToKeyframe = loopSchedulesAndData.imagesToKeyframe

        if 'img2img_fix_steps' in opts.data and opts.data["img2img_fix_steps"]:  # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
            opts.data["img2img_fix_steps"] = False
        if scheduled_clipskip is not None:
            opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
        if scheduled_noise_multiplier is not None:
            opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
        if scheduled_ddim_eta is not None:
            opts.data["eta_ddim"] = scheduled_ddim_eta
        if scheduled_ancestral_eta is not None:
            opts.data["eta_ancestral"] = scheduled_ancestral_eta

        if mode == '3D' and (cmd_opts.lowvram or cmd_opts.medvram):
            if predict_depths: depth_model.to('cpu')
            devices.torch_gc()
            lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            sd_hijack.model_hijack.hijack(sd_model)

        #_/‾𝓟𝓻𝓮𝓿_𝓲𝓶𝓰‾𝓢𝓮𝓬𝓽𝓲𝓸𝓷‾𝟐‾‾‾‾‾\__________

        # prepare for 4 possible generate() calls, for optical flow redo, redo generation, actual generation, and special 1st frame redo generation
        gen_args = (args, keys, anim_args, loop_args, controlnet_args, root)
        gen_kwargs = {'frame': frame_idx, 'sampler_name': scheduled_sampler_name}

        # if history has at least one entry, prev_img will be available
        if prev_img is not None:
            # intercept before generation and override to grayscale
            if anim_args.color_force_grayscale in ['Before/After', 'Before']:
                prev_img = bgr2gray_bgr(prev_img)

            # store seed
            seed_store = args.seed
            args.use_init = True

            # diffusion_redo, redo generations N times
            if int(anim_args.diffusion_redo) > 0:
                root.init_sample = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))
                prev_img = redo_generation(np.copy(prev_img), cc_sample, cc_alpha, *gen_args, **gen_kwargs)

            # optical flow generation, before generation
            if anim_args.optical_flow_redo_generation != 'None':
                root.init_sample = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))
                prev_img = optical_flow_generation(np.copy(prev_img), cc_sample, cc_alpha, redo_flow_factor, raft_model, *gen_args, **gen_kwargs)

            # restore seed
            args.seed = seed_store

            # set args.use_init and set the RGB PIL root.init_sample using the BGR opencv_image
            args.use_init = True
            root.init_sample = Image.fromarray(cv2.cvtColor(np.copy(prev_img), cv2.COLOR_BGR2RGB))

        #_/‾𝓡𝓖𝓑‾𝓖𝓮𝓷𝓮𝓻𝓪𝓽𝓲𝓸𝓷‾‾‾‾‾\_______________
        
        # disp before/after generation turns off/on the webui preview mechanism in favor of our own, and pauses our display timer to prevent it from showing the generation
        disp.before_generation()
        
        image = generate(*gen_args, **gen_kwargs)
        if image is None: break
        
        disp.after_generation()

        # back to BGR for the other operations
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # back to BGR for after generation procedures

        #‾\_𝓑𝓖𝓡_𝓐𝓯𝓽𝓮𝓻_𝓖𝓮𝓷𝓮𝓻𝓪𝓽𝓲𝓸𝓷_____/‾‾‾‾‾‾‾‾

        # special diffusion_redo for color coherence on frame 0, only if cc behavior is 'before generation' anbd cc source is image/video
        # extra generation uses actual generation as if it were a prev_img to be color matched
        if frame_idx == 0 and anim_args.color_coherence != 'None' and anim_args.color_coherence_source in ['Image Path', 'Video Init', 'Video Path'] \
                          and anim_args.color_coherence_behavior in ['Before', 'Before/After'] and cc_sample is not None:
            opencv_image = redo_generation(np.copy(opencv_image), cc_sample, cc_alpha, *gen_args, **gen_kwargs, redo_for_cc_before_gen=True)

        # hybrid video compositing: After generation
        if anim_args.hybrid_composite == 'After Generation':
            opencv_image, hybrid_overlay = hybrid_composite(args, anim_args, frame_idx, np.copy(opencv_image), depth_model, hybrid_comp_schedules, inputframes_path, hybridframes_path, root,
                                                            conform=hybrid_comp_conform, raft_model=raft_model, return_iterations=return_conform_iterations)

        # hybrid video motion: After generation, flow from current frame to next frame (skips last frame)
        if turbo_steps == 1 and anim_args.hybrid_motion_behavior == 'After Generation' and anim_args.hybrid_motion != 'None' and img_history.length >= 1 and frame_idx < anim_args.max_frames:
            opencv_image, prev_motion = hybrid_motion(frame_idx, np.copy(opencv_image), img_history.get_by_key_or_none(frame_idx-1),
                                                      hybrid_motion_history.get_by_key_or_none(frame_idx-1), args, anim_args,
                                                      inputfiles, hybridframes_path, hybrid_flow_factor, raft_model, matrix_flow=matrix_flow)

        # apply transforms to opencv_image, get depth, matrix flow, translation z for matrix flow
        if anim_args.animation_behavior == 'After Generation' and turbo_steps == 1:
            opencv_image, depth, matrix_flow, translation_z_estimate = anim_frame_warp(np.copy(opencv_image), anim_args, keys, frame_idx, depth_model=depth_model, depth=None,
                                                                                       device=root.device, half_precision=root.half_precision, inputfiles=inputfiles, hybridframes_path=hybridframes_path,
                                                                                       prev_flow=hybrid_motion_history.get_by_key_or_none(frame_idx-1),
                                                                                       prev_img=img_history.get_by_key_or_none(frame_idx-1),
                                                                                       raft_model=raft_model, z_translation_list=z_translation_history.list_states())
            z_translation_history.add_state(translation_z_estimate)

        if anim_args.color_coherence != 'None':
            # get color match sample from 1st generation - if source Image/Video Init, color match sample should exist already.
            if cc_sample is None:
                cc_sample = np.copy(opencv_image)
                print(f"Color match captured from frame {frame_idx}")

            # color coherence - after generation - reinforcing the pre-generation match if set on Before/After
            if anim_args.color_coherence_behavior in ['Before/After', 'After'] and cc_alpha > 0 and cc_sample is not None:
                opencv_image = color_matcher.maintain_colors(np.copy(opencv_image), cc_sample, cc_alpha, anim_args.color_coherence, frame_idx, console_msg="after generation", cc_mix_outdir=cc_mix_outdir, timestring=root.timestring)

        # force grayscale after generation - force grayscale after the color match, and before overlay
        if anim_args.color_force_grayscale in ['Before/After', 'After']:
            opencv_image = bgr2gray_bgr(np.copy(opencv_image))

        # hybrid comp overlay mask (before final overlay mask)
        if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_do_overlay_mask != 'None' and hybrid_overlay is not None:
            overlay_args = (args, anim_args, np.copy(opencv_image), frame_idx, inputframes_path, maskframes_path)
            overlay_kwargs = {'mask_override': hybrid_overlay} if anim_args.hybrid_use_init_image else {'video_mask_override': hybrid_overlay}
            overlay_kwargs.update({'invert_override': anim_args.hybrid_comp_mask_do_overlay_mask != 'Overlay'})
            opencv_image = do_overlay_mask(*overlay_args, **overlay_kwargs)

        # overlay mask last before opencv image is added to turbo_history
        if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
            opencv_image = do_overlay_mask(args, anim_args, opencv_image.astype(np.uint8), frame_idx, inputframes_path, maskframes_path)
       
        # convert for compatibility
        opencv_image = opencv_image.astype(np.uint8)

        # all modes use turbo_history for single image or double image cadence output
        turbo_history.add_state(opencv_image, frame_idx)

        # store current image as prev_img in first loop
        if prev_img is None: prev_img = np.copy(opencv_image)

        # loop compositing if cadence 1
        if turbo_steps == 1:
            prev_img = make_composite_with_conform(loop_comp_type, np.copy(opencv_image), np.copy(prev_img), loop_comp_alpha, anim_args.loop_comp_conform_method, raft_model=raft_model, 
                                                   conform=loop_comp_conform, order=anim_args.loop_comp_order, iterations=anim_args.loop_comp_conform_iterations)

        #_/‾𝓒𝓪𝓭𝓮𝓷𝓬𝓮‾‾‾‾‾\_______________________

        ''' 📺 Preview mechanism is also triggered and buffered from cadence 
            Cadence uses turbo_history, where each history state key is the frame index from when it was created
            • Whether turbo_mode is 1 or 2, removes (pops) the newest frame from history 
            • 💪 All frames get saved in cadence 💾 
                • Turbo mode 1 saves the frame at it's frame index immediately and adds to history, without additional processing
                    • frame 0 uses turbo mode 1 and outputs immediately, becoming the turbo_prev_image for the next cadence index
                • Turbo mode 2 uses the frame as the turbo_next image and gets it's next_idx
                    • turbo_prev and prev_idx are set using img_history by frame_idx
                    • prev_idx/next_idx are spaced apart by turbo_steps and they define the cadence index loop
                    • The cadence loop animates turbo_prev & turbo_next images into two independent sets of prev and next images
                      which can be thought of like two unmixed filmstrips, each the length of turbo_steps in frames.
                      Processes happen on the unmixed strips and only at the end does it mix them into single frames & save.
        '''
        
        # turbo_mode 1: single image cadence saves the 1 image, updates history and preview
        #            2: double image cadence does turbo_step tweening cycles between 2 images, updating history and preview
        turbo_mode = 1 if turbo_steps == 1 else 2

        # history is keyed to frame indexes. So, the rest of this code is based on thie turbo_next_image's next_idx
        # pop turbo next image from history: pop image, frame index from oldest item in turbo history
        turbo_next_image, next_idx = turbo_history.pop()

        # optional feature: extra video if in double cadence, you may save turbo_history images as they are popped as 'turbo_{timestring}_{frame_idx}.png'. These are never the same later, so get em while they're hot.
        if anim_args.cadence_save_turbo_frames:
            cadence_save('turbo_' + root.timestring, 0 if next_idx == 0 else int(next_idx/turbo_steps), outdir_turbo, turbo_next_image, \
                         anim_args.save_depth_maps, depth_model, depth, anim_args.midas_weight, cmd_opts, lowvram, sd_hijack, sd_model, devices, root)

        # if single image cadence, or double cadence on frame index 0, save image + store histories immediately
        if turbo_mode == 1 or next_idx == 0:
            cadence_save(root.timestring, next_idx, args.outdir, turbo_next_image, anim_args.save_depth_maps, depth_model, depth, anim_args.midas_weight, \
                         cmd_opts, lowvram, sd_hijack, sd_model, devices, root)
            img_history.add_state(turbo_next_image.astype(np.uint8), next_idx)
            hybrid_motion_history.add_state(prev_motion, next_idx)

            # webui preview display when not in cadence or on first frame
            disp.add_to_display_queue(bgr_np2pil_rgb(turbo_next_image))

        # double image cadence - uses img history to retrieve the last turbo_next_image and next_idx and uses as turbo_prev_image, prev_idx
        # frame 0 was the first turbo_next_image and it was saved immediately with no tweening. Now, we continue it's tween into the next turbo_next_image.
        # Example: At cadence 5, the tween steps if linear easing are [0.2, 0.4, 0.6, 0.8, 1], always starting past the last one and ending fully at 1.
        if turbo_mode == 2 and next_idx > 0:
            # get last turbo_next_image (now altered), the last next_idx from img history and use as turbo_prev_image, prev_idx
            last_next_idx = next_idx - turbo_steps
            turbo_prev_image = img_history.get_by_key(last_next_idx)
            prev_idx = last_next_idx + 1

            # start/prev index is one frame ahead of last cadence end
            start_idx = prev_idx

            # end_idx is for range functions like the for loop which require the end to be +1
            end_idx = start_idx + turbo_steps

            # do once for setup in-loop
            print(f"\033[94mCadence setup for tween between frames:\033[0m {start_idx} to {end_idx-1} ")

            # erase last_turbo before passing it again and making another last_turbo (or they nest)
            if 'last_turbo' in turbo:
                del turbo['last_turbo']

            depth = None

            # build turbo dict
            turbo = build_turbo_dict(turbo, turbo_prev_image, turbo_next_image, start_idx, end_idx, keys, img_history.get_by_key_or_none(last_next_idx), hybrid_motion_history.get_by_key_or_none(last_next_idx))

            # run animation
            turbo, translation_z_estimates = cadence_animation_frames(turbo, depth, depth_model, inputfiles, hybridframes_path, anim_args, keys, root, z_translation_list=z_translation_history.list_states())
            for item in translation_z_estimates:
                z_translation_history.add_state(item) 

            # hybrid compositing before motion
            if anim_args.hybrid_composite == 'Before Motion' and anim_args.hybrid_comp_conform_method != 'None':
                turbo = cadence_hybrid_comp(turbo, inputframes_path, dimensions, args, anim_args, raft_model)

            # run hybrid motion 
            if anim_args.hybrid_motion != 'None':
                turbo = cadence_hybrid_motion(turbo, inputfiles, inputframes_path, hybridframes_path, args, anim_args, raft_model)

            # hybrid compositing after motion
            if anim_args.hybrid_composite in ['Normal', 'After Generation'] and anim_args.hybrid_comp_conform_method != 'None':
                turbo = cadence_hybrid_comp(turbo, inputframes_path, dimensions, args, anim_args, raft_model)

            # run temporal flow 
            turbo = cadence_temporal_flow(turbo, anim_args, raft_model)

            # run morphological transformations
            turbo = cadence_morpho(turbo, anim_args, raft_model)

            # run cadence flow after all other operations
            turbo = cadence_flow(turbo, anim_args.optical_flow_cadence, raft_model, updown_scale)
        
            # _/‾𝓒𝓪𝓭𝓮𝓷𝓬𝓮‾𝓛𝓸𝓸𝓹‾‾‾‾‾\_________________

            # cadence_idx loop inside outer frame_idx loop
            for cadence_idx in range(start_idx, end_idx):
                # update ui progress during cadence
                state.job = f"frame {cadence_idx + 1}/{anim_args.max_frames+1}"
                state.job_no = cadence_idx + 1
                if state.skipped:
                    print("\n** PAUSED **")
                    state.skipped = False
                    while not state.skipped:
                        time.sleep(0.1)
                    print("** RESUMING **")

                # retrieve this frame's images and depth from pre-calculated turbo dict
                t = turbo[cadence_idx]
                turbo_prev, turbo_next, depth, ckey = t['prev'], t['next'], t['depth'], t['ckey']
                tween = ckey['tween']

                # srt files inside cadence
                if opts.data.get("deforum_save_gen_info_as_srt"):
                    params_string = format_animation_params(keys, prompt_series, cadence_idx)
                    write_frame_subtitle(srt_filename, cadence_idx, srt_frame_duration, f"F#: {cadence_idx}; Cadence: {tween['prime'] < 1.0}; Seed: {args.seed}; {params_string}")
                    params_string = None

                #_/‾𝓒𝓪𝓭𝓮𝓷𝓬𝓮‾𝓑𝓵𝓮𝓷𝓭‾‾‾‾‾\________________

                # on 1st frame of cadence, the turbo_next is not advanced | on last frame of cadence turbo_next is tween 1 and fully shown
                if tween['prime'] < 1.0:
                    img = turbo_prev * (1.0 - tween['image']) + turbo_next * tween['image']
                else: # tween['prime'] == 1 (last cadence image)
                    img = turbo_next

                # make image compatible with the rest of the operations below
                img = img.astype(np.uint8)

                #‾\_𝓐𝓯𝓽𝓮𝓻_𝓒𝓪𝓭𝓮𝓷𝓬𝓮 𝓑𝓵𝓮𝓷𝓭_____/‾‾‾‾‾‾‾‾‾

                # color match after generation
                if anim_args.color_coherence != 'None':
                    if anim_args.color_coherence_in_cadence and anim_args.color_coherence_behavior not in ['Before'] and anim_args.color_coherence_source in ['Video Init', 'Video Path'] and ckey['cc_alpha'] != 0:
                        # get cc_sample for video on every frame
                        cc_sample = get_cc_sample_from_video_frame(anim_args.color_coherence_source, args.outdir, cc_vid_folder, get_frame_name(cc_vid_path), cadence_idx, dimensions)
                    
                    # color coherence - in cadence if after generation - very useful for bitmap or grayscale direct morpho output
                    if cc_sample is not None:
                        img = color_matcher.maintain_colors(img, cc_sample, ckey['cc_alpha'], anim_args.color_coherence, cadence_idx, suppress_console=True, console_msg="in cadence after generation", cc_mix_outdir=cc_mix_outdir, timestring=root.timestring)

                # intercept img after color match and override to grayscale
                if anim_args.color_force_grayscale in ['Before/After', 'After']:
                    img = bgr2gray_bgr(img)

                # hybrid comp overlay mask (before final overlay mask)
                if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_do_overlay_mask and hybrid_overlay is not None:
                    overlay_args = (args, anim_args, img, cadence_idx, inputframes_path, maskframes_path)
                    overlay_kwargs = {'mask_override': hybrid_overlay} if anim_args.hybrid_use_init_image else {'video_mask_override': hybrid_overlay}
                    overlay_kwargs.update({'invert_override': anim_args.hybrid_comp_mask_do_overlay_mask != 'Overlay'})
                    img = do_overlay_mask(*overlay_args, **overlay_kwargs)

                # overlay mask last before prev_imgs are created
                if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
                    img = do_overlay_mask(args, anim_args, img, cadence_idx, inputframes_path, maskframes_path)

                # save img and store necessary histories
                cadence_save(root.timestring, cadence_idx, args.outdir, img, anim_args.save_depth_maps, depth_model, depth, anim_args.midas_weight, cmd_opts, lowvram, sd_hijack, sd_model, devices, root)
                img_history.add_state(img.astype(np.uint8), cadence_idx)
                hybrid_motion_history.add_state(prev_motion, cadence_idx)
                
                # Even though the generation doesn't happen here, we flag it as the one NOT tween frame, for it to time the cycle properly. 
                disp.add_to_display_queue(bgr_np2pil_rgb(img))

                # processed finished turbo_next_image becomes turbo_prev_image for next cycle
                if cadence_idx == end_idx-1:
                    turbo_prev_image, prev_idx = np.copy(img), end_idx

                # image iteration loop - store current image as prev_img with optional compositing
                prev_img = make_composite_with_conform(ckey['loop_comp_type'], np.copy(img), np.copy(prev_img), ckey['loop_comp_alpha'], anim_args.loop_comp_conform_method, raft_model=raft_model,
                                                       conform=ckey['loop_comp_conform'], order=anim_args.loop_comp_order, iterations=anim_args.loop_comp_conform_iterations)

                # console log cadence frame creation with cadence flow type, tween_prime, tween_image, tween_flow
                print_cadence_msg(cadence_idx, anim_args.optical_flow_cadence, ckey, tween, anim_args.max_frames)

            del(turbo_prev, turbo_next, ckey, tween)
            gc.collect()

            #‾\_𝓔𝓷𝓭_𝓒𝓪𝓭𝓮𝓷𝓬𝓮_𝓛𝓸𝓸𝓹_____/​‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

        # report done to console, and whether image was saved now, or stored for cadence.
        if frame_idx <= anim_args.max_frames:
            print(f"\033[94mFrame complete:\033[0m {frame_idx}")

        # frame 0 adjusted for, images created, console reported complete, so advance index by turbo_steps
        frame_idx += turbo_steps 

        # advance the seed
        args.seed = next_seed(args, root)

        #‾\_𝓔𝓷𝓭_𝓕𝓻𝓪𝓶𝓮_𝓛𝓸𝓸𝓹_____/‾‾‾‾‾‾‾‾‾‾‾‾‾

    # remove things from vram
    if predict_depths and not keep_in_vram:
        depth_model.delete_model()  # handles adabins too

    if load_raft:
        raft_model.delete_model()

    # stop disp timer and restore automatic1111 live preview setting
    disp.stop()
    opts.live_previews_enable = live_previews_enable_store
