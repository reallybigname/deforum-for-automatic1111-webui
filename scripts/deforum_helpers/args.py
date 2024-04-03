import json
import os
import tempfile
import time
import cv2
from numpy import zeros
from types import SimpleNamespace
import modules.paths as ph
import modules.shared as sh
from modules.sd_samplers import samplers_for_img2img
from modules.processing import get_fixed_seed
from .defaults import get_guided_imgs_default_json
from .deforum_controlnet import controlnet_component_names
from .general_utils import get_os, substitute_placeholders
from .composite import make_composite

def RootArgs():
    return {
        "device": sh.device,
        "models_path": ph.models_path + '/Deforum',
        "half_precision": not sh.cmd_opts.no_half,
        "clipseg_model": None,
        "mask_preset_names": ['everywhere', 'video_mask'],
        "frames_cache": [],
        "raw_batch_name": None,
        "raw_seed": None,
        "timestring": "",
        "subseed": -1,
        "subseed_strength": 0,
        "seed_internal": 0,
        "init_sample": None,
        "noise_mask": None,
        "initial_info": None,
        "first_frame": None,
        "animation_prompts": None,
        "current_user_os": get_os(),
        "tmp_deforum_run_duplicated_folder": os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    }

# 'Midas-3.1-BeitLarge' is temporarily removed until fixed. Can add it back anytime as it's supported in the back-end depth code
def DeforumAnimArgs():
    optical_flow_choices_normal = ['DIS UltraFast', 'DIS Fast', 'DIS Medium', 'DIS Slow', 'DIS Fine', 'DIS UltraFine', 'Farneback', 'RAFT']
    optical_flow_choices_contrib = ['DeepFlow', 'DenseRLOF', 'DualTVL1', 'PCAFlow', 'SF']
    optical_flow_choices = optical_flow_choices_normal
    
    # IN PROGRESS testing nvidia flows
    # optical_flow_choices_nvidia = ['NvidiaOpticalFlowBM', 'NvidiaOpticalFlowDual_TVL1', 'NvidiaOpticalFlowPyrLK', 'NvidiaOpticalFlowFarneback', 'NvidiaOpticalFlowDeepFlow']
    # optical_flow_choices += optical_flow_choices_nvidia

    # test for presence of cv2.optflow to see if we can add the contrib options for flow
    if hasattr(cv2, 'optflow'):
        # also test for one of the optical flow types specific to optflow object, in case optflow is still defined without containing the flow functions
        if hasattr(cv2.optflow, 'calcOpticalFlowDenseRLOF'): 
            optical_flow_choices += optical_flow_choices_contrib
    optical_flow_choices_with_none = ['None'] + optical_flow_choices[:]
    optical_flow_choices_morpho = optical_flow_choices_with_none[:] + ['No Flow (Direct)']
    composite_mask_lists = make_composite()
    
    anim_args = {
        "animation_mode": {
            "label": "Animation mode",
            "type": "radio",
            "choices": ['2D', '3D', 'Video Input', 'Interpolation'],
            "value": "2D",
            "info": "set animation mode • hides non-relevant params upon change"
        },
        "max_frames": {
            "label": "Max frames",
            "type": "number",
            "precision": 0,
            "value": 120,
            "info": "end at this frame number",
        },
        "border": {
            "label": "2D Padding mode",
            "type": "radio",
            "choices": ['reflect101', 'wrap', 'replicate'],
            "value": "reflect101",
            "info": "generation method for new edge pixels • hover selection for info"
        },
        "updown_scale": {
            "label": "Up/downscale warp/animation operations",
            "type": "radio",
            "choices": ['None', '1.5x', '2x', '3x', '4x'],
            "value": "None",
            "info": "Upscales, processes image warping, downscales - slower"
        },
        "animation_behavior": {
            "label": "Animation behavior",
            "type": "radio",
            "choices": ['Normal', 'After Generation'],
            "value": "Normal",
            "info": "normal is before generation. after generation is mostly for controlnet when using cadence 1"
        },
        "angle": {
            "label": "Angle° ⭮",
            "type": "textbox",
            "value": "0:(0)",
            "info": "2D image [🞤|⚊] • [⭮|⭯] rotate [RIGHT|LEFT] in degrees°"
        },
        "zoom": {
            "label": "Zoom 🔍",
            "type": "textbox",
            "value": "0:(1.0025+0.002*sin(1.25*3.14*t/30))",
            "info": "2D image [🞤|⚊] • [🗚|🗛] zooms [IN|OUT] • (static=1.00) (multiplier)"
        },
        "translation_x": {
            "label": "Translation X 🡄🡆",
            "type": "textbox",
            "value": "0:(0)",
            "info": "3D viewport [🞤|⚊] • 2D image [⚊|🞤] • moves [LEFT|RIGHT] on X axis [🡄|🡆]"
        },
        "translation_y": {
            "label": "Translation Y 🡇🡅",
            "type": "textbox",
            "value": "0:(0)",
            "info": "3D viewport [🞤|⚊] • 2D image [⚊|🞤] • moves [UP|DOWN] on Y axis  [🡅|🡇]"
        },
        "translation_z": {
            "label": "Translation Z ⇱⇲",
            "type": "textbox",
            "value": "0:(1.75)",
            "info": "3D viewport [🞤|⚊] • moves [IN|OUT] on Z axis (speed affected by FOV°) [⇱|⇲]"
        },
        "transform_center_x": {
            "label": "Transform Center X 🡄⯐🡆",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": "2D image [⚊|🞤] • moves [LEFT|RIGHT] center of X axis for angle°/zoom (center=0.5) [🡄|🡆]"
        },
        "transform_center_y": {
            "label": "Transform Center Y 🡅⯐🡇",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": "2D image [⚊|🞤] • moves [UP|DOWN] center of Y axis for angle°/zoom (center=0.5) [🡅|🡇]"
        },
        "rotation_3d_x": {
            "label": "Rotation 3D X ⮍⚊⮏",
            "type": "textbox",
            "value": "0:(0)",
            "info": "3D viewport [🞤|⚊] • pitch [UP|DOWN] by degrees° on X axis (lateral) [⮍⚊⮏]"
        },
        "rotation_3d_y": {
            "label": "Rotation 3D Y ⮎|⮌",
            "type": "textbox",
            "value": "0:(0)",
            "info": "3D viewport [⚊|🞤] • yaw [LEFT|RIGHT] by degrees° on Y axis (vertical) [⮎|⮌]"
        },
        "rotation_3d_z": {
            "label": "Rotation 3D Z ⭮/⭯",
            "type": "textbox",
            "value": "0:(0)",
            "info": "3D viewport [🞤|⚊] • roll [RIGHT|LEFT] by degrees° on Z axis (longitudinal) [⭮/⭯]"
        },
        "enable_perspective_flip": {
            "label": "Enable perspective flip",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "perspective_flip_theta": {
            "label": "Perspective flip theta",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "perspective_flip_phi": {
            "label": "Perspective flip phi",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "perspective_flip_gamma": {
            "label": "Perspective flip gamma",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "perspective_flip_fv": {
            "label": "Perspective flip tv",
            "type": "textbox",
            "value": "0:(53)",
            "info": "the 2D vanishing point of perspective (rec. range 30-160)"
        },
        "noise_schedule": {
            "label": "Noise schedule",
            "type": "textbox",
            "value": "0:(0.065)",
            "info": ""
        },
        "strength_schedule": {
            "label": "Strength schedule",
            "type": "textbox",
            "value": "0:(0.65)",
            "info": "(0-1) similarity to previous frame/init image | also reduces steps taken to [steps*(1-strength)]"
        },
        "contrast_schedule": {
            "label": "Contrast schedule",
            "type": "textbox",
            "value": "0:(1.0)",
            "interactive": True,
            "info": "adjusts the overall contrast per frame [neutral at 1.0, recommended to *not* play with this param]"
        },
        "cfg_scale_schedule": {
            "label": "CFG scale schedule",
            "type": "textbox",
            "value": "0:(7)",
            "info": "how closely the image should conform to the prompt. Lower values produce more creative results. (recommended range 5-15)`"
        },
        "enable_steps_scheduling": {
            "label": "Enable steps scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "steps_schedule": {
            "label": "Steps schedule",
            "type": "textbox",
            "value": "0:(25)",
            "info": "mainly allows using more than 200 steps. otherwise, it's a mirror-like param of 'strength schedule'"
        },
        "fov_schedule": {
            "label": "FOV schedule",
            "type": "textbox",
            "value": "0:(70)",
            "info": "adjusts the scale at which the canvas is moved in 3D by the translation_z value. [maximum range -180 to +180, with 0 being undefined. Values closer to 180 will make the image have less depth, while values closer to 0 will allow more depth]"
        },
        "aspect_ratio_schedule": {
            "label": "Aspect ratio schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "adjusts the aspect ratio (of the depth calculations only)"
        },
        "aspect_ratio_use_old_formula": {
            "label": "Use old aspect ratio formula",
            "type": "checkbox",
            "value": False,
            "info": "for backward compatibility. uses the formula: `width/height`"
        },
        "near_schedule": {
            "label": "Near schedule",
            "type": "textbox",
            "value": "0:(200)",
            "info": ""
        },
        "far_schedule": {
            "label": "Far schedule",
            "type": "textbox",
            "value": "0:(10000)",
            "info": ""
        },
        "seed_schedule": {
            "label": "Seed schedule",
            "type": "textbox",
            "value": '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)',
            "info": ""
        },
        "pix2pix_img_cfg_scale_schedule": {
            "label": "Pix2Pix img CFG schedule",
            "type": "textbox",
            "value": "0:(1.5)",
            "info": "ONLY in use when working with a P2P ckpt!"
        },
        "enable_subseed_scheduling": {
            "label": "Enable Subseed scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "subseed_schedule": {
            "label": "Subseed schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "subseed_strength_schedule": {
            "label": "Subseed strength schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "enable_sampler_scheduling": {
            "label": "Enable sampler scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "sampler_schedule": {
            "label": "Sampler schedule",
            "type": "textbox",
            "value": '0: ("Euler a")',
            "info": "allows keyframing different samplers. Use names as they appear in ui dropdown in 'run' tab"
        },
        "use_noise_mask": {
            "label": "Use noise mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_schedule": {
            "label": "Mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "noise_mask_schedule": {
            "label": "Noise mask schedule",
            "type": "textbox",
            "value": '0: ("{video_mask}")',
            "info": ""
        },
        "enable_checkpoint_scheduling": {
            "label": "Enable checkpoint scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "checkpoint_schedule": {
            "label": "allows keyframing different sd models. use *full* name as appears in ui dropdown",
            "type": "textbox",
            "value": '0: ("model1.ckpt"), 100: ("model2.safetensors")',
            "info": "allows keyframing different sd models. use *full* name as appears in ui dropdown"
        },
        "enable_clipskip_scheduling": {
            "label": "Enable CLIP skip scheduling",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "clipskip_schedule": {
            "label": "CLIP skip schedule",
            "type": "textbox",
            "value": "0:(2)",
            "info": "",
            "visible": False
        },
        "enable_noise_multiplier_scheduling": {
            "label": "Enable noise multiplier scheduling",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "noise_multiplier_schedule": {
            "label": "Noise multiplier schedule",
            "type": "textbox",
            "value": "0:(1.05)",
            "info": ""
        },
        "resume_from_timestring": {
            "label": "Resume from timestring",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "resume_timestring": {
            "label": "Resume timestring",
            "type": "textbox",
            "value": "20230129210106",
            "info": ""
        },
        "enable_ddim_eta_scheduling": {
            "label": "Enable DDIM ETA scheduling",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "noise multiplier; higher = more unpredictable results"
        },
        "ddim_eta_schedule": {
            "label": "DDIM ETA Schedule",
            "type": "textbox",
            "value": "0:(0)",
            "visible": False,
            "info": ""
        },
        "enable_ancestral_eta_scheduling": {
            "label": "Enable Ancestral ETA scheduling",
            "type": "checkbox",
            "value": False,
            "info": "noise multiplier; applies to Euler a and other samplers that have the letter 'a' in them"
        },
        "ancestral_eta_schedule": {
            "label": "Ancestral ETA Schedule",
            "type": "textbox",
            "value": "0:(1)",
            "visible": False,
            "info": ""
        },
        "amount_schedule": {
            "label": "Amount schedule",
            "type": "textbox",
            "value": "0:(0.1)",
            "info": ""
        },
        "kernel_schedule": {
            "label": "Kernel schedule",
            "type": "textbox",
            "value": "0:(5)",
            "info": ""
        },
        "sigma_schedule": {
            "label": "Sigma schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": ""
        },
        "threshold_schedule": {
            "label": "Threshold schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": ""
        },
        "color_coherence": {
            "label": "Color coherence",
            "type": "dropdown",
            "choices": ['None', 'LAB', 'HSV', 'RGB', 'HM', 'Reinhard', 'MVGD', 'MKL', 'HM-MVGD-HM', 'HM-MKL-HM'],
            "value": "LAB",
            "info": "method for color match w/sample"
        },
        "color_coherence_source": {
            "label": "Color coherence source",
            "type": "dropdown",
            "choices": ['First Frame', 'Image Path', 'Video Init', 'Video Path'],
            "value": "First Frame",
            "info": "choose a source for the color match sampling"
        },
        "color_coherence_behavior": {
            "label": "Color coherence behavior",
            "type": "dropdown",
            "choices": ['Before/After', 'Before', 'After'],
            "value": "Before/After",
            "info": "whether color matching happens before and/or after generation"
        },
        "color_coherence_in_cadence": {
            "label": "Color coherence in cadence",
            "type": "checkbox",
            "value": False,
            "info": "color matches in cadence if behavior contains 'After'"
        },        
        "color_coherence_image_path": {
            "label": "Color coherence image path",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/I1.png",
            "info": "path to an image to use as the color match",
            "visible": False
        },
        "color_coherence_video_path": {
            "label": "Color coherence video path",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/V1.mp4",
            "info": "path to a video to use as the color match",
            "visible": False
        },
        "color_coherence_video_from_to_nth": {
            "label": "Color coherence video from|to|nth",
            "type": "textbox",
            "value": "0|-1|1",
            "info": "frame #'s: from|to|every nth, all frames=0|-1|1",
            "visible": False
        },
        "color_force_grayscale": {
            "label": "Color force Grayscale",
            "type": "dropdown",
            "choices": ['None', 'Before/After', 'Before', 'After'],
            "value": "None",
            "info": "force frames to be in grayscale before and/or after generation"
        },
        "color_coherence_alpha_schedule": {
            "label": "Color coherence alpha schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "alpha level (0-1) for color coherence each frame (allows drift)"
        },
        "loop_comp_type_schedule": {
            "label": "Loop composite filter schedule",
            "type": "textbox",
            "value": '0: ("none")',
            "info": "compositing type for this frame that becomes the previous frame"
        },
        "loop_comp_alpha_schedule": {
            "label": "Loop composite alpha schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "alpha level (0-1) of compositing applied to new image in forward mode, or old image in backward mode"
        },
        "loop_comp_conform_method": {
            "label": "Loop conform flow alignment method",
            "type": "dropdown",
            "choices": optical_flow_choices_with_none,
            "value": "None",
            "info": "method of optical flow used to align the image composited with the image compositing"
        },
        "loop_comp_conform_schedule": {
            "label": "Loop conform factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "alignment is achieved with 1 - but other values are possible",
            "visible": False
        },
        "loop_comp_conform_iterations": {
            "label": "Loop conform iterations",
            "type": "slider",
            "minimum": 1,
            "maximum": 10,
            "step": 1,
            "value": 1,
            "interactive": True,
            "info": "number of iterations to alignment using structural similarity",
            "visible": False
        },
        "loop_comp_order": {
            "label": "Loop composite order",
            "type": "radio",
            "choices": ['Forward', 'Backward'],
            "value": "Forward",
            "info": "[forward: new A⇾B old | backward: old A⇾B new] reverses layer order for compositing & alpha mix"
        },
        "morpho_flow": {
            "label": "Morphological flow",
            "type": "dropdown",
            "choices": optical_flow_choices_morpho,
            "value": "None",
            "info": "select an optical flow type to enable the morphological schedule"
        },
        "morpho_flow_factor_schedule": {
            "label": "Morphological flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "factor -/+: -1=reverse, 0=none, 1=full, etc.",
            "visible": False
        },
        "morpho_schedule": {
            "label": "Morphological schedule",
            "type": "textbox",
            "value": '0: ("dilate|rect")',
            "info": "make things dilate, erode, etc - see accordion for help",
            "visible": False
        },
        "morpho_iterations_schedule": {
            "label": "Morphological iteration schedule",
            "type": "textbox",
            "value": '0: (0)',
            "info": "(integer) # of iterations for morphological schedule (0=disabled)",
            "visible": False
        },
        "morpho_image_type": {
            "label": "Morphological image type",
            "type": "radio",
            "choices": ['Grayscale', '3-Color', 'Bitmap'],
            "value": "Grayscale",
            "info": "The way the image is passed to morphology. 3-color does 3 channels separately.",
            "visible": False
        },
        "morpho_bitmap_threshold": {
            "label": "Morphological bitmap threshold",
            "type": "textbox",
            "value": "127|255",
            "info": "bitmap threshold for b&w image for morphology » Low '0-255|0-255' High",
            "visible": False
        },
        "morpho_cadence_behavior": {
            "label": "Morphological behavior inside cadence",
            "type": "radio",
            "choices": ['None', 'Forward', 'Bounce'],
            "value": "None",
            "info": "special morphological behaviors for cadence, forward or bounce mode",
            "visible": False
        },
        "temporal_flow": {
            "label": "Temporal flow",
            "type": "dropdown",
            "choices": optical_flow_choices_with_none,
            "value": "None",
            "info": "warps image with optical flow from last img"
        },
        "temporal_flow_return_target": {
            "label": "Temporal flow return image",
            "type": "radio",
            "choices": ['Image', 'Target'],
            "value": "Image",
            "info": "whether to return the image or the target history image, applying flow from target to image",
            "visible": False
        },
        "temporal_flow_target_frame_schedule": {
            "label": "Temporal flow target frame schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "[integer|0+|default:1] schedule of Nth history frame to reference for optical flow. 0th frame is the initial state of the frame from this cycle. 1st frame is the last cycle.",
            "visible": False
        },
        "temporal_flow_factor_schedule": {
            "label": "Temporal flow factor schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": "factor -/+: -1=reverse, 0=none, 1=full, etc.",
            "visible": False
        },
        "temporal_flow_cadence_behavior": {
            "label": "Temporal flow behavior inside cadence",
            "type": "radio",
            "choices": ['None', 'Forward', 'Bounce'],
            "value": "None",
            "info": "special temporal flow behaviors for cadence, forward or bounce mode",
            "visible": False
        },
        "temporal_flow_motion_stabilizer_factor_schedule": {
            "label": "Temporal flow motion stabilizer factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "factor -/+: 1=stabilized frame, 0=off, -1=opposite, etc.",
            "visible": False
        },
        "temporal_flow_rotation_stabilizer_factor_schedule": {
            "label": "Temporal flow rotation stabilizer factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "factor -/+: 1=stabilized frame, 0=off, -1=opposite, etc.",
            "visible": False
        },
        "optical_flow_redo_generation": {
            "label": "Optical flow generation",
            "type": "dropdown",
            "choices": optical_flow_choices_with_none,
            "value": "None",
            "info": "generates diffusion, gets optical flow from prev diffusion to new diffusion, applies flow to new diffusion, uses as init image for generation (takes 2x as long)"
        },
        "redo_flow_factor_schedule": {
            "label": "Generation flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "factor -/+: -1=reverse, 0=none, 1=normal, etc.",
            "visible": False
        },
        "diffusion_redo": {
            "label": "Redo generation",
            "type": "slider",
            "minimum": 0,
            "maximum": 50,
            "step": 1,
            "value": 0,
            "interactive": True,
            "info": "this option renders N times before the final render. it is suggested to lower your steps if you up your redo. seed is randomized during redo generations and restored afterwards"
        },
        "diffusion_cadence": {
            "label": "Cadence",
            "type": "slider",
            "minimum": 1,
            "maximum": 100,
            "step": 1,
            "value": 2,
            "interactive": True,
            "info": "creates N-1 frames in-between diffusions"
        },
        "cadence_diffusion_easing_schedule": {
            "label": "Cadence easing schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": "[range -1/+1] | -1=easing out/in | 0=linear | +1=easing in/out"
        },
        "optical_flow_cadence": {
            "label": "Cadence flow",
            "type": "dropdown",
            "choices": optical_flow_choices_with_none,
            "value": "None",
            "info": "adds morphing to the blend of in-between cadence images"
        },
        "cadence_flow_easing_schedule": {
            "label": "Cadence flow easing schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": "[range -1/+1] | -1=easing out/in | 0=linear | +1=easing in/out",
            "visible": False
        },
        "cadence_flow_factor_schedule": {
            "label": "Cadence flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "factor -/+: -1=reverse, 0=none, 1=normal, etc.",
            "visible": False
        },
        "cadence_flow_warp_factor_schedule": {
            "label": "Cadence flow warp factor schedule",
            "type": "textbox",
            "value": '0:("[1]")',
            "info": "Step distance between A/B image states that flow covers during a cadence cycle.",
            "visible": False
        },
        "cadence_save_turbo_frames": {
            "label": "Save 'turbo' frames",
            "type": "checkbox",
            "value": False,
            "info": "saves source frames from cadence as 'turbo_{timestring}_{index}.png'",
            "visible": False
        },
        "noise_type": {
            "label": "Noise type",
            "type": "radio",
            "choices": ['uniform', 'perlin'],
            "value": "perlin",
            "info": ""
        },
        "perlin_w": {
            "label": "Perlin W",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_h": {
            "label": "Perlin H",
            "type": "slider",
            "minimum": 0.1,
            "maximum": 16,
            "step": 0.1,
            "value": 8,
            "visible": False
        },
        "perlin_octaves": {
            "label": "Perlin octaves",
            "type": "slider",
            "minimum": 1,
            "maximum": 7,
            "step": 1,
            "value": 4
        },
        "perlin_persistence": {
            "label": "Perlin persistence",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.02,
            "value": 0.5
        },
        "use_depth_warping": {
            "label": "Use depth warping",
            "type": "checkbox",
            "value": True,
            "info": "whether to load depth model for 3D depth warping"
        },
        "depth_algorithm": {
            "label": "Depth Algorithm",
            "type": "dropdown",
            "choices": ['Midas+AdaBins (old)', 'Zoe+AdaBins (old)', 'Midas-3-Hybrid', 'AdaBins', 'Zoe', 'Leres'],
            "value": "Midas-3-Hybrid",
            "info": "choose a depth algorithm for 3D mode"
        },
        "midas_weight": {
            "label": "MiDaS/Zoe weight",
            "type": "number",
            "precision": None,
            "value": 0.2,
            "info": "sets a midpoint at which a depth-map is to be drawn: range [-1 to +1]",
            "visible": False
        },
        "padding_mode": {
            "label": "3D Padding mode",
            "type": "radio",
            "choices": ['border', 'reflection', 'zeros'],
            "value": "border",
            "info": "controls the 3D handling of pixels outside the field of view as they come into the scene",
            "visible": False
        },
        "sampling_mode": {
            "label": "Sampling mode",
            "type": "radio",
            "choices": ['bicubic', 'bilinear', 'nearest'],
            "value": "bicubic",
            "info": "controls sampling quality"
        },
        "save_depth_maps": {
            "label": "Save 3D depth maps",
            "type": "checkbox",
            "value": False,
            "info": "save animation's depth maps as extra files"
        },
        "video_init_path": {
            "label": "Video init path/URL",
            "type": "textbox",
            "value": 'https://deforum.github.io/a1/V1.mp4',
            "info": "file path or URL to a video file"
        },
        "extract_nth_frame": {
            "label": "Extract nth frame",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "only extract every Nth frame of video. note: changes frame rate"
        },
        "extract_from_frame": {
            "label": "Extract from frame",
            "type": "number",
            "precision": 0,
            "value": 0,
            "info": "starting frame index for video extraction"
        },
        "extract_to_frame": {
            "label": "Extract to frame",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "ending frame index for video extraction (-1 = entire video)"
        },
        "overwrite_extracted_frames": {
            "label": "Overwrite extracted frames",
            "type": "checkbox",
            "value": False,
            "info": "whether to overwrite previously extracted frames in inputframes folder"
        },
        "use_mask_video": {
            "label": "Use mask video",
            "type": "checkbox",
            "value": False,
            "info": "if checked, make sure you have a valid video mask path"
        },
        "video_mask_path": {
            "label": "Video mask path",
            "type": "textbox",
            "value": 'https://deforum.github.io/a1/VM1.mp4',
            "info": "video masks obey controls of the mask init tab"
        },
        "hybrid_generate_inputframes": {
            "label": "Generate inputframes",
            "type": "checkbox",
            "value": False,
            "scale": 1,
            "info": "extract video init path (first step for compositing or motion!)"
        },
        "hybrid_comp_save_extra_frames":{
            "label": "Save extra frames",
            "type": "checkbox",
            "value": False,
            "scale": 1,
            "info": "Save `hybridframes` for debugging compositing and motion"
        },
        "reallybigname_css_btn": {
            "value": "Inject reallybigname css",
            "type": "button",
            "size": "sm",
            "scale": 1,
            "elem_classes": "rbn_css_btn"
        },
        "hybrid_generate_human_masks": {
            "label": "Generate human masks",
            "type": "radio",
            "choices": ['None', 'PNGs', 'Video', 'Both'],
            "value": "None",
            "info": "not automatic yet! Make a 'Video' mask by starting generation. Once extraction finishes and it starts frame 0, cancel job, use the mask video in your `Video mask path`."
        },
        "hybrid_motion": {
            "label": "Hybrid motion",
            "type": "dropdown",
            "choices": ['None', 'Matrix Flow', 'Optical Flow', 'Perspective', 'Affine'], # choice of 'Matrix Flow' can be added but is experimental
            "value": "None",
            "info": "Optical flow tracks all pixels. Perspective/Affine are RANSAC camera tracking. Matrix flow is experimental, alters your actual motion path based on matrix and uses flow for movement in the frame."
        },
        "hybrid_flow_method": {
            "label": "Flow method",
            "type": "dropdown",
            "choices": optical_flow_choices,
            "value": "RAFT",
            "info": "different forms of optical flow capture methods",
            "visible": False
        },
        "hybrid_flow_factor_schedule": {
            "label": "Flow factor schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "factor -/+: -1=reverse, 0=none, 1=normal, etc.",
            "visible": False
        },
        "hybrid_motion_behavior": {
            "label": "Hybrid motion behavior",
            "type": "radio",
            "choices": ['Before Generation', 'After Generation'],
            "value": "Before Generation",
            "info": "whether to apply motion to init image or to image after generation",
            "visible": False
        },
        "hybrid_motion_use_prev_img": {
            "label": "Motion use prev img",
            "type": "checkbox",
            "value": False,
            "info": "gets motion from last frame's generated image to current video image, rather than just using video images",
            "visible": False
        },
        "hybrid_flow_consistency": {
            "label": "Flow consistency mask",
            "type": "checkbox",
            "value": False,
            "info": "masks the edges of moving items to clean up unreliable flows",
            "visible": False
        },
        "hybrid_consistency_blur": {
            "label": "Consistency mask blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 16,
            "step": 1,
            "value": 2,
            "visible": False,
            "info": "blur the consistency mask which determines where the flow is applied"
        },
        # "hybrid_video_init_flow_amount": {
        #     "label": "Video Init Flow Amount (needs Generate Inputframes)",
        #     "type": "slider",
        #     "minimum": 0,
        #     "maximum": 1,
        #     "step": 0.01,
        #     "value": 1
        # },
        # "hybrid_cn1_flow_amount": {
        #     "label": "CN1 Flow Amount",
        #     "type": "slider",
        #     "minimum": 0,
        #     "maximum": 1,
        #     "step": 0.01,
        #     "value": 0
        # },
        # "hybrid_cn2_flow_amount": {
        #     "label": "CN2 Flow Amount",
        #     "type": "slider",
        #     "minimum": 0,
        #     "maximum": 1,
        #     "step": 0.01,
        #     "value": 0
        # },
        # "hybrid_cn3_flow_amount": {
        #     "label": "CN3 Flow Amount",
        #     "type": "slider",
        #     "minimum": 0,
        #     "maximum": 1,
        #     "step": 0.01,
        #     "value": 0
        # },
        # "hybrid_cn4_flow_amount": {
        #     "label": "CN4 Flow Amount",
        #     "type": "slider",
        #     "minimum": 0,
        #     "maximum": 1,
        #     "step": 0.01,
        #     "value": 0
        # },
        # "hybrid_cn5_flow_amount": {
        #     "label": "CN5 Flow Amount",
        #     "type": "slider",
        #     "minimum": 0,
        #     "maximum": 1,
        #     "step": 0.01,
        #     "value": 0
        # },
        "hybrid_composite": {
            "label": "Hybrid composite",
            "type": "radio",
            "choices": ['None', 'Normal', 'Before Motion', 'After Generation'],
            "value": "None",
            "info": "video mixed into init image (normal or before motion) or image (after generation)",
        },
        "hybrid_comp_conform_method": {
            "label": "Force conform optical flow method",
            "type": "dropdown",
            "choices": optical_flow_choices_with_none,
            "value": "None",
            "info": "use optical flow to force the conformation of video to image or image to video or somewhere in between",
            "visible": False
        },
        "hybrid_comp_conform_iterations": {
            "label": "Force conform iterations",
            "type": "slider",
            "minimum": 1,
            "maximum": 10,
            "step": 1,
            "value": 1,
            "info": "number of iterations to alignment using structural similarity",
            "visible": False
        },
        "hybrid_comp_conform_schedule": {
            "label": "Force conform value",
            "type": "textbox",
            "value": "0:(1)",
            "info": "[0-1] forces conformation of shapes: 0=video conforms to image | 0.5=both conform half way to each other | 1=image conforms to video",
            "visible": False
        },
        "hybrid_use_init_image": {
            "label": "Use init image as video",
            "type": "checkbox",
            "value": False,
            "info": "allows hybrid compositing w/only init image instead of video",
            "visible": False
        },
        "hybrid_use_first_frame_as_init_image": {
            "label": "First frame as init image",
            "type": "checkbox",
            "value": True,
            "info": "if inputframes are generated, automatically make the first frame the init image",
            "visible": True
        },
        "hybrid_comp_type": {
            "label": "Composite type",
            "type": "dropdown",
            "choices": ['None'] + composite_mask_lists['color'],
            "value": "None",
            "info": "compositing type with mix controlled by compositing alpha",
            "visible": False
        },
        "hybrid_comp_alpha_schedule": {
            "label": "Composite alpha schedule",
            "type": "textbox",
            "value": "0:(0.5)",
            "info": "range 0-1 | hybrid compositing alpha is the master mix of any hybrid operation you're doing.",
            "visible": False
        },
        "hybrid_comp_mask_type": {
            "label": "Composite mask type",
            "type": "dropdown",
            "choices": ['None', 'Depth', 'Video Depth'] + composite_mask_lists['grayscale'],
            "value": "None",
            "info": "mask type for creating 'composite mask' using image and video: 'depth' from image | 'video depth' from video",
            "visible": False
        },
        "hybrid_comp_mask_inverse": {
            "label": "Composite mask inverse",
            "type": "checkbox",
            "value": False,
            "info": "invert the composite mask",
            "visible": False
        },
        "hybrid_comp_mask_auto_contrast": {
            "label": "Composite mask auto-contrast",
            "type": "checkbox",
            "value": False,
            "info": "composite mask auto-contrast within low/high schedules",
            "visible": False
        },
        "hybrid_comp_mask_equalize": {
            "label": "Composite mask equalize",
            "type": "radio",
            "choices": ['None', 'Before', 'After', 'Both'],
            "value": "None",
            "info": "equalize mask before or after auto-contrast operation (or both)",
            "visible": False
        },
        "hybrid_comp_mask_do_overlay_mask": {
            "label": "Use composite mask as overlay mask",
            "type": "radio",
            "choices": ['None', 'Overlay', 'Invert Overlay'],
            "value": "None",
            "info": "creates overlay mask using composite mask, applied before normal overlay mask, uses settings from overlay mask",
            "visible": False
        },
        "hybrid_comp_mask_alpha_schedule": {
            "label": "Composite mask alpha schedule",
            "type": "textbox",
            "value": "0:(1.0)",
            "info": "affects hybrid comp mask's alpha (not comp alpha)",
            "elem-classes": "section_full_width",
            "visible": False
        },
        "hybrid_comp_mask_contrast_schedule": {
            "label": "Composite mask contrast schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "|default 1] contrast multiplier for any hybrid comp mask",
            "elem-classes": "section_full_width",
            "visible": False
        },
        "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {
            "label": "Composite mask auto contrast cutoff high schedule",
            "type": "textbox",
            "value": "0:(1)",
            "info": "[0-1] If using auto-contrast, this is high cutoff percentage (default 1)",
            "elem-classes": "section_full_width",
            "visible": False
        },
        "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {
            "label": "Composite mask auto contrast cutoff low schedule",
            "type": "textbox",
            "value": "0:(0)",
            "info": "[0-1 ]If using auto-contrast, this is low cutoff percentage (default 0)",
            "elem-classes": "section_full_width",
            "visible": False
        }
    }
    # modifies "type": "textbox"
    # if not an entry, add placeholder containing value
    #                  add max_lines of 6
    for key, value in anim_args.items():
        if isinstance(value, dict) and value.get("type") == "textbox":
            if "placeholder" not in value:
                value["placeholder"] = value["value"]
            if "lines" not in value:
                value["lines"] = 1
            if "max_lines" not in value:
                value["max_lines"] = 10
    return anim_args

def DeforumArgs():
    return {
        "W": {
            "label": "Width",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 64,
            "value": 512,
        },
        "H": {
            "label": "Height",
            "type": "slider",
            "minimum": 64,
            "maximum": 2048,
            "step": 64,
            "value": 512,
        },
        "show_info_on_ui": True,
        "tiling": {
            "label": "Tiling",
            "type": "checkbox",
            "value": False,
            "info": "enable for seamless-tiling of each generated image. Experimental"
        },
        "restore_faces": {
            "label": "Restore faces",
            "type": "checkbox",
            "value": False,
            "info": "enable to trigger webui's face restoration on each frame during the generation"
        },
        "seed_resize_from_w": {
            "label": "Resize seed from width",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed_resize_from_h": {
            "label": "Resize seed from height",
            "type": "slider",
            "minimum": 0,
            "maximum": 2048,
            "step": 64,
            "value": 0,
        },
        "seed": {
            "label": "Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Starting seed for the animation. -1 for random"
        },
        "sampler": {
            "label": "Sampler",
            "type": "dropdown",
            "choices": [x.name for x in samplers_for_img2img],
            "value": samplers_for_img2img[0].name,
        },
        "steps": {
            "label": "Step",
            "type": "slider",
            "minimum": 1,
            "maximum": 200,
            "step": 1,
            "value": 25,
        },
        "batch_name": {
            "label": "Batch name",
            "type": "textbox",
            "value": "Deforum_{timestring}",
            "info": "output images will be placed in a folder with this name ({timestring} token will be replaced) inside the img2img output folder. Supports params placeholders. e.g {seed}, {w}, {h}, {prompts}"
        },
        "seed_behavior": {
            "label": "Seed behavior",
            "type": "radio",
            "choices": ['iter', 'fixed', 'random', 'ladder', 'alternate', 'schedule'],
            "value": "iter",
            "info": "controls the seed behavior that is used for animation. hover on the options to see more info"
        },
        "seed_iter_N": {
            "label": "Seed iter N",
            "type": "number",
            "precision": 0,
            "value": 1,
            "info": "for how many frames the same seed should stick before iterating to the next one"
        },
        "use_init": {
            "label": "Use init",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "strength": {
            "label": "strength",
            "type": "slider",
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "value": 0.8,
        },
        "strength_0_no_init": {
            "label": "Strength 0 no init",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "init_image": {
            "label": "Init image",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/I1.png",
            "info": ""
        },
        "use_mask": {
            "label": "Use mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "use_alpha_as_mask": {
            "label": "Use alpha as mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_file": {
            "label": "Mask file",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/M1.jpg",
            "info": ""
        },
        "invert_mask": {
            "label": "Invert mask",
            "type": "checkbox",
            "value": False,
            "info": ""
        },
        "mask_contrast_adjust": {
            "label": "Mask contrast adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "mask_brightness_adjust": {
            "label": "Mask brightness adjust",
            "type": "number",
            "precision": None,
            "value": 1.0,
            "info": ""
        },
        "overlay_mask": {
            "label": "Overlay mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "mask_overlay_blur": {
            "label": "Mask overlay blur",
            "type": "slider",
            "minimum": 0,
            "maximum": 64,
            "step": 1,
            "value": 4,
        },
        "fill": {
            "label": "Mask fill",
            "type": "radio",
            "radio_type": "index",
            "choices": ['fill', 'original', 'latent noise', 'latent nothing'],
            "value": 'original',
            "info": ""
        },
        "full_res_mask": {
            "label": "Full res mask",
            "type": "checkbox",
            "value": True,
            "info": ""
        },
        "full_res_mask_padding": {
            "label": "Full res mask padding",
            "type": "slider",
            "minimum": 0,
            "maximum": 512,
            "step": 1,
            "value": 4,
        },
        "reroll_blank_frames": {
            "label": "Reroll blank frames",
            "type": "radio",
            "choices": ['reroll', 'interrupt', 'ignore'],
            "value": "ignore",
            "info": "what to do with blank frames (from glitches or NSFW filter being turned on): reroll with +1 seed, interrupt the animation generation, or do nothing"
        },
        "reroll_patience": {
            "label": "Reroll patience",
            "type": "number",
            "precision": None,
            "value": 10,
            "info": ""
        },
    }

def LoopArgs():
    return {
        "use_looper": {
            "label": "Enable guided images mode",
            "type": "checkbox",
            "value": False,
        },
        "init_images": {
            "label": "Images to use for keyframe guidance",
            "type": "textbox",
            "lines": 9,
            "value": get_guided_imgs_default_json(),
        },
        "image_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox",
            "value": "0:(0.75)"
        },
        "blendFactorMax": {
            "label": "Blend factor max",
            "type": "textbox",
            "value": "0:(0.35)",
        },
        "blendFactorSlope": {
            "label": "Blend factor slope",
            "type": "textbox",
            "value": "0:(0.25)",
        },
        "tweening_frames_schedule": {
            "label": "Tweening frames schedule",
            "type": "textbox",
            "value": "0:(20)"
        },
        "color_correction_factor": {
            "label": "Color correction factor",
            "type": "textbox",
            "value": "0:(0.075)",
        }
    }

def ParseqArgs():
    return {
        "parseq_manifest": {
            "label": "Parseq Manifest (JSON or URL)",
            "type": "textbox",
            "lines": 4,
            "value": None,
        },
        "parseq_use_deltas": {
            "label": "Use delta values for movement parameters",
            "type": "checkbox",
            "value": True,
        }
    }

def DeforumOutputArgs():
    return {
        "skip_video_creation": {
            "label": "Skip video creation",
            "type": "checkbox",
            "value": False,
            "info": "If enabled, only images will be saved"
        },
        "fps": {
            "label": "FPS",
            "type": "slider",
            "minimum": 1,
            "maximum": 240,
            "step": 1,
            "value": 15,
        },
        "make_gif": {
            "label": "Make GIF",
            "type": "checkbox",
            "value": False,
            "info": "make gif in addition to the video/s"
        },
        "delete_imgs": {
            "label": "Delete Imgs",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready"
        },
        "image_path": {
            "label": "Image path",
            "type": "textbox",
            "value": "C:/SD/20230124234916_%09d.png",
        },
        "add_soundtrack": {
            "label": "Add soundtrack",
            "type": "radio",
            "choices": ['None', 'File', 'Video Init'],
            "value": "None",
            "info": "add audio to video from file/url or video init"
        },
        "soundtrack_path": {
            "label": "Soundtrack path",
            "type": "textbox",
            "value": "https://deforum.github.io/a1/A1.mp3",
            "info": "abs. path or url to audio file"
        },
        "r_upscale_video": {
            "label": "Upscale",
            "type": "checkbox",
            "value": False,
            "info": "upscale output imgs when run is finished"
        },
        "r_upscale_factor": {
            "label": "Upscale factor",
            "type": "dropdown",
            "choices": ['x2', 'x3', 'x4'],
            "value": "x2",
        },
        "r_upscale_model": {
            "label": "Upscale model",
            "type": "dropdown",
            "choices": ['realesr-animevideov3', 'realesrgan-x4plus', 'realesrgan-x4plus-anime'],
            "value": 'realesr-animevideov3',
        },
        "r_upscale_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": True,
            "info": "don't delete upscaled imgs",
        },
        "store_frames_in_ram": {
            "label": "Store frames in ram",
            "type": "checkbox",
            "value": False,
            "info": "auto-delete imgs when video is ready",
            "visible": False
        },
        "frame_interpolation_engine": {
            "label": "Engine",
            "type": "radio",
            "choices": ['None', 'RIFE v4.6', 'FILM'],
            "value": "None",
            "info": "select the frame interpolation engine. hover on the options for more info"
        },
        "frame_interpolation_x_amount": {
            "label": "Interp X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_slow_mo_enabled": {
            "label": "Slow Mo",
            "type": "checkbox",
            "value": False,
            "visible": False,
            "info": "Slow-Mo the interpolated video, audio will not be used if enabled",
        },
        "frame_interpolation_slow_mo_amount": {
            "label": "Slow-Mo X",
            "type": "slider",
            "minimum": 2,
            "maximum": 10,
            "step": 1,
            "value": 2,
        },
        "frame_interpolation_keep_imgs": {
            "label": "Keep Imgs",
            "type": "checkbox",
            "value": False,
            "info": "Keep interpolated images on disk",
            "visible": False
        },
        "frame_interpolation_use_upscaled": {
            "label": "Use Upscaled",
            "type": "checkbox",
            "value": False,
            "info": "Interpolate upscaled images, if available",
            "visible": False
        }
    }

def get_component_names():
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts', 'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(), *LoopArgs().keys(), *controlnet_component_names()]

def get_settings_component_names():
    return [name for name in get_component_names()]

def pack_args(args_dict, keys_function):
    return {name: args_dict[name] for name in keys_function()}

def process_args(args_dict_main, run_id):
    from .settings import load_args
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    p = args_dict_main['p']

    root = SimpleNamespace(**RootArgs())
    args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumArgs()})
    anim_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumAnimArgs()})
    video_args = SimpleNamespace(**{name: args_dict_main[name] for name in DeforumOutputArgs()})
    parseq_args = SimpleNamespace(**{name: args_dict_main[name] for name in ParseqArgs()})
    loop_args = SimpleNamespace(**{name: args_dict_main[name] for name in LoopArgs()})
    controlnet_args = SimpleNamespace(**{name: args_dict_main[name] for name in controlnet_component_names()})

    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    args_loaded_ok = True
    if override_settings_with_file:
        args_loaded_ok = load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, video_args, custom_settings_file, root, run_id)

    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    negative_prompts = negative_prompts.replace('--neg', '')  # remove --neg from negative_prompts if received by mistake
    root.animation_prompts = {key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}" for key, val in root.animation_prompts.items()}

    if args.seed == -1:
        root.raw_seed = -1
    args.seed = get_fixed_seed(args.seed)
    if root.raw_seed != -1:
        root.raw_seed = args.seed
    root.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    if not args.use_init and not anim_args.hybrid_use_init_image:
        args.init_image = None

    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    current_arg_list = [args, anim_args, video_args, parseq_args, root]
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args
