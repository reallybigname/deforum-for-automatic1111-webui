import gc
import os
import gradio as gr
import modules.paths as ph
from .general_utils import get_os
from .upscaling import process_ncnn_upscale_vid_upload_logic
from .video_audio_utilities import extract_number, get_quick_vid_info, get_ffmpeg_params
from .frame_interpolation import process_interp_vid_upload_logic, process_interp_pics_upload_logic, gradio_f_interp_get_fps_and_fcount
from .vid2depth import process_depth_vid_upload_logic

f_models_path = ph.models_path + '/Deforum'

def handle_change_functions(l_vars):
    # interpolation
    for output in [l_vars['color_force_grayscale'], l_vars['noise_tab_column']]:
        l_vars['animation_mode'].change(fn=disable_by_interpolation, inputs=l_vars['animation_mode'], outputs=output)

    # run
    l_vars['sampler'].change(fn=show_when_ddim, inputs=l_vars['sampler'], outputs=l_vars['enable_ddim_eta_scheduling'])
    l_vars['sampler'].change(fn=show_when_ancestral_samplers, inputs=l_vars['sampler'], outputs=l_vars['enable_ancestral_eta_scheduling'])
    l_vars['enable_ancestral_eta_scheduling'].change(fn=hide_if_false, inputs=l_vars['enable_ancestral_eta_scheduling'], outputs=l_vars['ancestral_eta_schedule'])
    l_vars['enable_ddim_eta_scheduling'].change(fn=hide_if_false, inputs=l_vars['enable_ddim_eta_scheduling'], outputs=l_vars['ddim_eta_schedule'])

    # resume expander
    l_vars['resume_from_timestring'].change(fn=expand_if_checked, inputs=l_vars['resume_from_timestring'], outputs=l_vars['resume_and_batch_accord'])

    # keyframes
    l_vars['override_settings_with_file'].change(fn=hide_if_false, inputs=l_vars['override_settings_with_file'], outputs=l_vars['custom_settings_file'])   
   
    # max frames switching
    max_frames_visibility_inputs = [l_vars['animation_mode'], l_vars['hybrid_composite'], l_vars['hybrid_use_init_image']]
    l_vars['animation_mode'].change(fn=change_max_frames_visibility, inputs=max_frames_visibility_inputs, outputs=l_vars['max_frames_wrapper'])

    # 2D3D: diffusion cadence, guided images, temporal_flow_row, morpho_row, optical_flow_generation_row
    show_if_2D3D_outputs = [l_vars['guided_images_accord'], l_vars['temporal_flow_row'], l_vars['morpho_row'], l_vars['optical_flow_generation_row']]
    for output in show_if_2D3D_outputs:
        l_vars['animation_mode'].change(fn=show_if_2D3D, inputs=l_vars['animation_mode'], outputs=output)

    # 2D
    two_d_related_outputs = [l_vars['only_2d_motion_column'], l_vars['border']]
    for output in two_d_related_outputs:
        l_vars['animation_mode'].change(fn=enable_2d_related_stuff, inputs=l_vars['animation_mode'], outputs=output)
    pers_flip_outputs = [l_vars['per_f_th_row'], l_vars['per_f_ph_row'], l_vars['per_f_ga_row'], l_vars['per_f_f_row']]
    for output in pers_flip_outputs:
        l_vars['enable_perspective_flip'].change(fn=hide_if_false, inputs=l_vars['enable_perspective_flip'], outputs=output)
        l_vars['animation_mode'].change(fn=per_flip_handle, inputs=[l_vars['animation_mode'], l_vars['enable_perspective_flip']], outputs=output)
    pers_flip_visibility = [l_vars['enable_per_f_row'], l_vars['both_anim_mode_motion_params_column']]
    for output in pers_flip_visibility:
        l_vars['animation_mode'].change(fn=disable_pers_flip_accord, inputs=l_vars['animation_mode'], outputs=output)

    # 3D
    three_d_related_outputs = [l_vars['padding_mode'], l_vars['only_3d_motion_column'],
                               l_vars['depth_warp_row_1'], l_vars['depth_warp_row_2'], l_vars['depth_warp_row_3'], l_vars['depth_warp_row_4'],
                               l_vars['depth_warp_row_5'], l_vars['depth_warp_row_6'], l_vars['depth_warp_row_7']]
    for output in three_d_related_outputs:
        l_vars['animation_mode'].change(fn=disble_3d_related_stuff, inputs=l_vars['animation_mode'], outputs=output)

    # clip skip
    l_vars['enable_clipskip_scheduling'].change(hide_if_false, inputs=l_vars['enable_clipskip_scheduling'], outputs=l_vars['clipskip_schedule'])

    # seed
    l_vars['seed_behavior'].change(fn=change_seed_iter_visibility, inputs=l_vars['seed_behavior'], outputs=l_vars['seed_iter_N_row'])
    l_vars['seed_behavior'].change(fn=change_seed_schedule_visibility, inputs=l_vars['seed_behavior'], outputs=l_vars['seed_schedule_row'])

    # hybrid section disabling
    hybrid_sections_2D3D = [l_vars['hybrid_video_compositing_section'], l_vars['humans_masking_row']]
    for output in hybrid_sections_2D3D:
        l_vars['animation_mode'].change(fn=show_if_2D3D, inputs=l_vars['animation_mode'], outputs=output)
    l_vars['animation_mode'].change(fn=show_if_not_2D3D, inputs=l_vars['animation_mode'], outputs=l_vars['hybrid_msg_html'])

    # coherence (color)
    cc_inputs = [l_vars[name] for name in ['color_coherence', 'color_coherence_source']]
    cc_outputs = [l_vars[name] for name in ['color_coherence_alpha_schedule', 'color_coherence_image_path', 'color_coherence_video_path',
                                            'color_coherence_video_from_to_nth', 'color_coherence_source','color_coherence_behavior']]
    for inp in cc_inputs: inp.change(fn=color_coherence_master, inputs=cc_inputs, outputs=cc_outputs)

    # cadence
    cadence_ins = [l_vars[name] for name in ['animation_mode', 'diffusion_cadence', 'optical_flow_cadence']]
    cadence_outs = [l_vars[name] for name in ['cadence_accord', 'cadence_save_turbo_frames', 'cadence_flow_easing_schedule', 'cadence_flow_factor_schedule',
                                              'cadence_flow_warp_factor_schedule', 'cadence_flow_warp_factor_schedule_html_row']]
    for inp in cadence_ins:
        inp.change(fn=cadence_master, inputs=cadence_ins, outputs=cadence_outs)
    # l_vars['diffusion_cadence'].input(fn=cadence_master, inputs=l_vars['diffusion_cadence'], outputs=l_vars['diffusion_cadence'])

    # noise
    l_vars['noise_type'].change(fn=change_perlin_visibility, inputs=l_vars['noise_type'], outputs=l_vars['perlin_row'])

    # depth warping
    l_vars['animation_mode'].change(fn=only_show_in_non_3d_mode, inputs=l_vars['animation_mode'], outputs=l_vars['depth_warp_msg_html'])
    l_vars['depth_algorithm'].change(fn=legacy_3d_mode, inputs=l_vars['depth_algorithm'], outputs=l_vars['midas_weight'])
    l_vars['depth_algorithm'].change(fn=show_leres_html_msg, inputs=l_vars['depth_algorithm'], outputs=l_vars['leres_license_msg'])

    l_vars['aspect_ratio_use_old_formula'].change(fn=hide_if_true, inputs=l_vars['aspect_ratio_use_old_formula'], outputs=l_vars['aspect_ratio_schedule'])

    # reallybigname css
    l_vars['reallybigname_css_btn'].click(fn=inject_reallybigname_css, inputs=l_vars['reallybigname_css_btn'], outputs=[gr.outputs.HTML(), l_vars['reallybigname_css_btn']])

    # hybrid composite
    hybrid_comp_outs = [l_vars[name] for name in \
        ['hybrid_use_init_image', 'hybrid_comp_mask_type', 'hybrid_comp_mask_equalize', 'hybrid_comp_mask_auto_contrast', 'hybrid_comp_mask_inverse',
         'hybrid_comp_mask_do_overlay_mask', 'hybrid_comp_type', 'hybrid_comp_alpha_schedule', 'hybrid_comp_mask_alpha_schedule', 'hybrid_comp_mask_contrast_schedule',
         'hybrid_comp_mask_auto_contrast_cutoff_high_schedule', 'hybrid_comp_mask_auto_contrast_cutoff_low_schedule', 'hybrid_comp_mask_schedule_rows', 'hybrid_comp_msg_row',
         'hybrid_comp_conform_method', 'hybrid_comp_conform_schedule', 'hybrid_comp_conform_iterations']]
    hybrid_comp_ins = [l_vars[name] for name in ['hybrid_composite', 'hybrid_comp_conform_method', 'hybrid_comp_mask_type']]
    for inp in hybrid_comp_ins:
        inp.change(fn=hybrid_comp_master, inputs=hybrid_comp_ins, outputs=hybrid_comp_outs)

    # hybrid motion
    hybrid_motion_outs = [l_vars[name] for name in ['hybrid_flow_method', 'hybrid_motion_behavior', 'hybrid_motion_use_prev_img', 'hybrid_flow_factor_schedule', \
                                                    'hybrid_flow_consistency', 'hybrid_consistency_blur']] # 'hybrid_motion_flow_amounts_row'
    hybrid_motion_ins = [l_vars[name] for name in ['hybrid_motion', 'hybrid_flow_method', 'hybrid_flow_consistency']]
    for inp in hybrid_motion_ins:
        inp.change(fn=hybrid_motion_master, inputs=hybrid_motion_ins, outputs=hybrid_motion_outs)

    # flow tools - morphological, temporal flow, optical flow generation
    morpho_inputs = [l_vars[name] for name in ['morpho_flow', 'morpho_image_type']]
    morpho_outputs = [l_vars[name] for name in ['morpho_flow_factor_schedule', 'morpho_schedule', 'morpho_iterations_schedule',
                                                'morpho_image_type', 'morpho_bitmap_threshold', 'morpho_cadence_behavior', 'morpho_help_wrapper']]
    for inp in morpho_inputs:
        inp.change(fn=morpho_master, inputs=morpho_inputs, outputs=morpho_outputs)

    # temporal flow
    # flow tools - morphological, temporal flow, optical flow generation
    temporal_flow_inputs = [l_vars[name] for name in ['temporal_flow']]
    temporal_flow_outputs = [l_vars[name] for name in ['temporal_flow_factor_schedule', 'temporal_flow_cadence_behavior', 'temporal_flow_return_target', 'temporal_flow_target_frame_schedule',
                                                       'temporal_flow_motion_stabilizer_factor_schedule', 'temporal_flow_rotation_stabilizer_factor_schedule']] 
    for inp in temporal_flow_inputs:
        inp.change(fn=temporal_flow_master, inputs=temporal_flow_inputs, outputs=temporal_flow_outputs)
           
    l_vars['temporal_flow'].change(fn=show_hide_if_none, inputs=l_vars['temporal_flow'], outputs=l_vars['temporal_flow_factor_schedule'])
    l_vars['temporal_flow'].change(fn=show_hide_if_none, inputs=l_vars['temporal_flow'], outputs=l_vars['temporal_flow_cadence_behavior'])

    # optical flow redo generation
    l_vars['optical_flow_redo_generation'].change(fn=show_hide_if_none, inputs=l_vars['optical_flow_redo_generation'], outputs=l_vars['redo_flow_factor_schedule'])

    # loop composite conform
    l_vars['loop_comp_conform_method'].change(fn=show_hide_if_none, inputs=l_vars['loop_comp_conform_method'], outputs=l_vars['loop_comp_conform_schedule'])
    l_vars['loop_comp_conform_method'].change(fn=show_hide_if_none, inputs=l_vars['loop_comp_conform_method'], outputs=l_vars['loop_comp_conform_iterations'])

    # output
    l_vars['fps'].change(fn=change_gif_button_visibility, inputs=l_vars['fps'], outputs=l_vars['make_gif'])
    l_vars['r_upscale_model'].change(fn=update_r_upscale_factor, inputs=l_vars['r_upscale_model'], outputs=l_vars['r_upscale_factor'])
    l_vars['ncnn_upscale_model'].change(fn=update_r_upscale_factor, inputs=l_vars['ncnn_upscale_model'], outputs=l_vars['ncnn_upscale_factor'])
    l_vars['ncnn_upscale_model'].change(update_upscale_out_res_by_model_name, inputs=[l_vars['ncnn_upscale_in_vid_res'], l_vars['ncnn_upscale_model']], outputs=l_vars['ncnn_upscale_out_vid_res'])
    l_vars['ncnn_upscale_factor'].change(update_upscale_out_res, inputs=[l_vars['ncnn_upscale_in_vid_res'], l_vars['ncnn_upscale_factor']], outputs=l_vars['ncnn_upscale_out_vid_res'])
    l_vars['vid_to_upscale_chosen_file'].change(vid_upscale_gradio_update_stats, inputs=[l_vars['vid_to_upscale_chosen_file'], l_vars['ncnn_upscale_factor']],
                                                outputs=[l_vars['ncnn_upscale_in_vid_fps_ui_window'], l_vars['ncnn_upscale_in_vid_frame_count_window'],
                                                         l_vars['ncnn_upscale_in_vid_res'], l_vars['ncnn_upscale_out_vid_res']])

    # output
    skip_video_creation_outputs = [l_vars['fps_out_format_row'], l_vars['soundtrack_row'], l_vars['store_frames_in_ram'], l_vars['make_gif'], l_vars['r_upscale_row'],
                                   l_vars['delete_imgs']]
    for output in skip_video_creation_outputs:
        l_vars['skip_video_creation'].change(fn=change_visibility_from_skip_video, inputs=l_vars['skip_video_creation'], outputs=output)
    l_vars['frame_interpolation_slow_mo_enabled'].change(fn=hide_if_false, inputs=l_vars['frame_interpolation_slow_mo_enabled'], outputs=l_vars['frame_interp_slow_mo_amount_column'])
    l_vars['frame_interpolation_engine'].change(fn=change_interp_x_max_limit, inputs=[l_vars['frame_interpolation_engine'], l_vars['frame_interpolation_x_amount']],
                                                  outputs=l_vars['frame_interpolation_x_amount'])
    # Populate the FPS and FCount values as soon as a video is uploaded to the FileUploadBox (vid_to_interpolate_chosen_file)
    l_vars['vid_to_interpolate_chosen_file'].change(gradio_f_interp_get_fps_and_fcount,
                                                      inputs=[l_vars['vid_to_interpolate_chosen_file'], l_vars['frame_interpolation_x_amount'], l_vars['frame_interpolation_slow_mo_enabled'],
                                                              l_vars['frame_interpolation_slow_mo_amount']],
                                                      outputs=[l_vars['in_vid_fps_ui_window'], l_vars['in_vid_frame_count_window'], l_vars['out_interp_vid_estimated_fps']])
    l_vars['vid_to_interpolate_chosen_file'].change(fn=hide_interp_stats, inputs=[l_vars['vid_to_interpolate_chosen_file']], outputs=[l_vars['interp_live_stats_row']])
    interp_hide_list = [l_vars['frame_interpolation_slow_mo_enabled'], l_vars['frame_interpolation_keep_imgs'], l_vars['frame_interpolation_use_upscaled'], l_vars['frame_interp_amounts_row'], l_vars['interp_existing_video_row']]
    for output in interp_hide_list:
        l_vars['frame_interpolation_engine'].change(fn=hide_interp_by_interp_status, inputs=l_vars['frame_interpolation_engine'], outputs=output)


# START gradio-to-frame-interoplation/ upscaling functions
def upload_vid_to_interpolate(file, engine, x_am, sl_enabled, sl_am, keep_imgs, in_vid_fps):
    # print msg and do nothing if vid not uploaded or interp_x not provided
    if not file or engine == 'None':
        return print("Please upload a video and set a proper value for 'Interp X'. Can't interpolate x0 times :)")
    f_location, f_crf, f_preset = get_ffmpeg_params()

    process_interp_vid_upload_logic(file, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps, f_models_path, file.orig_name)

def upload_pics_to_interpolate(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, fps, add_audio, audio_track):
    from PIL import Image

    if pic_list is None or len(pic_list) < 2:
        return print("Please upload at least 2 pics for interpolation.")
    f_location, f_crf, f_preset = get_ffmpeg_params()
    # make sure all uploaded pics have the same resolution
    pic_sizes = [Image.open(picture_path.name).size for picture_path in pic_list]
    if len(set(pic_sizes)) != 1:
        return print("All uploaded pics need to be of the same Width and Height / resolution.")

    resolution = pic_sizes[0]

    process_interp_pics_upload_logic(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, fps, f_models_path, resolution, add_audio, audio_track)

def ncnn_upload_vid_to_upscale(vid_path, in_vid_fps, in_vid_res, out_vid_res, upscale_model, upscale_factor, keep_imgs):
    if vid_path is None:
        print("Please upload a video :)")
        return
    f_location, f_crf, f_preset = get_ffmpeg_params()
    current_user = get_os()
    process_ncnn_upscale_vid_upload_logic(vid_path, in_vid_fps, in_vid_res, out_vid_res, f_models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user)

def upload_vid_to_depth(vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, depth_keep_imgs):
    # print msg and do nothing if vid not uploaded
    if not vid_to_depth_chosen_file:
        return print("Please upload a video :()")
    f_location, f_crf, f_preset = get_ffmpeg_params()

    process_depth_vid_upload_logic(vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth,
                                   vid_to_depth_chosen_file.orig_name, depth_keep_imgs, f_location, f_crf, f_preset, f_models_path)

# END gradio-to-frame-interpolation/ upscaling functions

# reallybigname UI helper functions

# class for managing true/false indexes (1-based for easy reference, so the 2nd element is 2)
class ValList:
    def __init__(self, val, n): # init list of length n with value
        self.lst = [val] * n
    def set(self, val, *true_indices): # set list indices as True (takes unlimited args rather than tuple or list)
        self.lst = [val if i+1 in true_indices else x for i, x in enumerate(self.lst)]
    def get_list(self): # retrieve list as 0-indexed list that it is
        return self.lst

def cadence_master(animation_mode, diffusion_cadence, optical_flow_cadence):
    vis = ValList(False, 6)
    if animation_mode in ['2D', '3D']:
        if diffusion_cadence > 1:
            vis.set(True, 1, 2)
            if optical_flow_cadence != 'None':
                vis.set(True, 3, 4, 5, 6)
    return tuple(gr.update(visible=v) for v in vis.get_list())

def color_coherence_master(color_coherence, color_coherence_source):
    vis = ValList(False, 6)
    if color_coherence != 'None':
        vis.set(True, 1, 5, 6)
        if color_coherence_source == 'Image Path':
            vis.set(True, 2)
        elif color_coherence_source == 'Video Path':
            vis.set(True, 3, 4)
    return tuple(gr.update(visible=v) for v in vis.get_list())

def hybrid_comp_master(hybrid_composite, hybrid_comp_conform_method, hybrid_composite_mask_type):
    vis = ValList(False, 17)
    if hybrid_composite != 'None':
        vis.set(True, 1, 2, 7, 8, 14, 15)
        if hybrid_comp_conform_method != 'None':
            vis.set(True, 16, 17)
        if hybrid_composite_mask_type != 'None':
            vis.set(True, 3, 4, 5, 6, 10, 11, 12, 13)
            if not hybrid_composite_mask_type in ['Depth', 'Video Depth']:
                vis.set(True, 9)
    return tuple(gr.update(visible=v) for v in vis.get_list())

def hybrid_motion_master(hybrid_motion, hybrid_flow_method, hybrid_flow_consistency):
    vis = ValList(False, 7)
    if hybrid_motion != 'None':
        vis.set(True, 2, 3)
        if hybrid_motion in ['Optical Flow', 'Matrix Flow']:
            vis.set(True, 1, 4, 5, 6)
        if hybrid_flow_consistency:
            vis.set(True, 7)
    return tuple(gr.update(visible=v) for v in vis.get_list())

def morpho_master(morpho_flow, morpho_image_type):
    vis = ValList(False, 7)
    if morpho_flow != 'None':
        vis.set(morpho_flow != 'No Flow (Direct)', 1)
        vis.set(True, 2, 3, 4)
        if morpho_image_type == 'Bitmap':
            vis.set(True, 5)
        vis.set(True, 6, 7)
    return tuple(gr.update(visible=v) for v in vis.get_list())

def temporal_flow_master(temporal_flow):
    vis = ValList(False, 6)
    if temporal_flow != 'None':
        vis.set(True, 1, 2, 3, 4, 5, 6)
    return tuple(gr.update(visible=v) for v in vis.get_list())

def expand_if_checked(b):
    return gr.update(open=b)

def show_hide_if_none(choice):
    return gr.update(visible=choice != 'None')

def hide_show_if_optical_flow(choice):
    return gr.update(visible=choice in ['Optical Flow', 'Matrix Flow'])

# UI helper functions
def change_visibility_from_skip_video(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)

def update_r_upscale_factor(choice):
    return gr.update(value='x4', choices=['x4']) if choice != 'realesr-animevideov3' else gr.update(value='x2', choices=['x2', 'x3', 'x4'])

def change_perlin_visibility(choice):
    return gr.update(visible=choice == "perlin")

def legacy_3d_mode(choice):
    return gr.update(visible=choice.lower() in ["midas+adabins (old)", 'zoe+adabins (old)'])

def not_legacy_3d_mode(choice):
    return gr.update(visible=choice.lower() not in ["midas+adabins (old)", 'zoe+adabins (old)'])

def change_seed_iter_visibility(choice):
    return gr.update(visible=choice == "iter")

def change_seed_schedule_visibility(choice):
    return gr.update(visible=choice == "schedule")

def disable_pers_flip_accord(choice):
    return gr.update(visible=True) if choice in ['2D', '3D'] else gr.update(visible=False)

def per_flip_handle(anim_mode, per_f_enabled):
    if anim_mode in ['2D', '3D'] and per_f_enabled:
        return gr.update(visible=True)
    return gr.update(visible=False)

def change_max_frames_visibility(mode_choice, hybrid_comp_choice, hybrid_use_image_choice):
    vis = not (mode_choice == 'Video Input' or (hybrid_comp_choice != 'None' and not hybrid_use_image_choice))
    return gr.update(visible=vis)

def change_diffusion_cadence_visibility(choice):
    return gr.update(visible=choice not in ['Video Input', 'Interpolation'])

def disble_3d_related_stuff(choice):
    return gr.update(visible=False) if choice != '3D' else gr.update(visible=True)

def only_show_in_non_3d_mode(choice):
    return gr.update(visible=False) if choice == '3D' else gr.update(visible=True)

def enable_2d_related_stuff(choice):
    return gr.update(visible=True) if choice == '2D' else gr.update(visible=False)

def disable_by_interpolation(choice):
    return gr.update(visible=False) if choice in ['Interpolation'] else gr.update(visible=True)

def disable_by_video_input(choice):
    return gr.update(visible=False) if choice in ['Video Input'] else gr.update(visible=True)

def change_gif_button_visibility(choice):
    if choice is None or choice == "":
        return gr.update(visible=True)
    return gr.update(visible=False, value=False) if int(choice) > 30 else gr.update(visible=True)

def hide_if_false(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)

def hide_if_true(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)

def disable_by_hybrid_composite_dynamic(choice, comp_mask_type):
    if choice in ['Normal', 'Before Motion', 'After Generation']:
        if comp_mask_type != 'None':
            return gr.update(visible=True)
    return gr.update(visible=False)


# Upscaling Gradio UI related funcs
def vid_upscale_gradio_update_stats(vid_path, upscale_factor):
    if not vid_path:
        return '---', '---', '---', '---'
    factor = extract_number(upscale_factor)
    fps, fcount, resolution = get_quick_vid_info(vid_path.name)
    in_res_str = f"{resolution[0]}*{resolution[1]}"
    out_res_str = f"{resolution[0] * factor}*{resolution[1] * factor}"
    return fps, fcount, in_res_str, out_res_str

def update_upscale_out_res(in_res, upscale_factor):
    if not in_res:
        return '---'
    factor = extract_number(upscale_factor)
    w, h = [int(x) * factor for x in in_res.split('*')]
    return f"{w}*{h}"

def update_upscale_out_res_by_model_name(in_res, upscale_model_name):
    if not upscale_model_name or in_res == '---':
        return '---'
    factor = 2 if upscale_model_name == 'realesr-animevideov3' else 4
    return f"{int(in_res.split('*')[0]) * factor}*{int(in_res.split('*')[1]) * factor}"

def hide_cadence_if_1(cadence_value):
    return gr.update(visible=True) if int(cadence_value) > 1 else gr.update(visible=False)

def hide_interp_by_interp_status(choice):
    return gr.update(visible=False) if choice == 'None' else gr.update(visible=True)

def change_interp_x_max_limit(engine_name, current_value):
    if engine_name == 'FILM':
        return gr.update(maximum=300)
    elif current_value > 10:
        return gr.update(maximum=10, value=2)
    return gr.update(maximum=10)

def hide_interp_stats(choice):
    return gr.update(visible=True) if choice is not None else gr.update(visible=False)

def show_if_not_2D3D(choice):
    return gr.update(visible=True) if choice not in ['2D', '3D'] else gr.update(visible=False)

def show_if_2D3D(choice):
    return gr.update(visible=True) if choice in ['2D', '3D'] else gr.update(visible=False)

def show_leres_html_msg(choice):
    return gr.update(visible=True) if choice.lower() == 'leres' else gr.update(visible=False)

def show_when_ddim(sampler_name):
    return gr.update(visible=True) if sampler_name.lower() == 'ddim' else gr.update(visible=False)

def show_when_ancestral_samplers(sampler_name):
    return gr.update(visible=True) if sampler_name.lower() in ['euler a', 'dpm++ 2s a', 'dpm2 a', 'dpm2 a karras', 'dpm++ 2s a karras'] else gr.update(visible=False)

def change_css(checkbox_status):
    if checkbox_status:
        display = "block"
    else:
        display = "none"

    html_template = f'''
        <style>
            #tab_deforum_interface .svelte-e8n7p6, #f_interp_accord {{
                display: {display} !important;
            }}
        </style>
        '''
    return html_template

def inject_reallybigname_css(css_path):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(current_folder)
    grandparent_folder = os.path.dirname(parent_folder)
    with open(grandparent_folder + '/css/reallybigname.css', 'r') as file:
        css_variable = file.read()
        css_variable = f'<style>{css_variable}</style>'
        gr.HTML(css_variable)
    return css_variable, gr.update(visible=False)
