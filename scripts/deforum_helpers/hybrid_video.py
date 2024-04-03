import os
import cv2
import pathlib
from .human_masking import video2humanmasks
from .video_audio_utilities import generate_inputframes, get_quick_vid_info

def hybrid_generation(args, anim_args, inputframes_path, hybrid_cn_paths, root):
    human_masks_path = os.path.join(args.outdir, 'human_masks')
    hybridframes_path = os.path.join(args.outdir, 'hybridframes')

    # create hybridframes folder if save extra frames option is enabled
    if anim_args.hybrid_comp_save_extra_frames:
        os.makedirs(hybridframes_path, exist_ok=True)
        # delete hybridframes if overwrite == true
        if anim_args.overwrite_extracted_frames and not anim_args.resume_from_timestring:
            delete_all_imgs_in_folder(hybridframes_path)

    # generate inputframes
    if anim_args.hybrid_generate_inputframes:
        video_fps = generate_inputframes(inputframes_path, anim_args.overwrite_extracted_frames,
                                         anim_args.video_init_path, anim_args.extract_nth_frame,
                                         anim_args.extract_from_frame, anim_args.extract_to_frame)
        
    # extract alpha masks of humans from the extracted input video imgs
    if anim_args.hybrid_generate_human_masks != "None":
        # create a folder for the human masks imgs to live in
        print(f"Checking /creating a folder for the human masks")
        os.makedirs(human_masks_path, exist_ok=True)
            
        # delete frames if overwrite = true
        if anim_args.overwrite_extracted_frames:
            delete_all_imgs_in_folder(human_masks_path)
        
        # in case that generate_input_frames isn't selected, we won't get the video fps rate as vid2frames isn't called, So we'll check the video fps in here instead
        if not anim_args.hybrid_generate_inputframes:
            _, video_fps, _ = get_quick_vid_info(anim_args.video_init_path)
            
        # calculate the correct fps of the masked video according to the original video fps and 'extract_nth_frame'
        output_fps = video_fps/anim_args.extract_nth_frame
        
        # generate the actual alpha masks from the input imgs
        print(f"Extracting alpha humans masks from the input frames")
        video2humanmasks(inputframes_path, human_masks_path, anim_args.hybrid_generate_human_masks, output_fps)
        
    # get sorted list of inputfiles
    inputfiles = sorted(pathlib.Path(inputframes_path).glob('*.jpg'))

    cn_files = []
    for path in hybrid_cn_paths:
        if path is None:
            cn_files.append(None)
        else:
            cn_files.append(sorted(pathlib.Path(path).glob('*.jpg')))

    if not anim_args.hybrid_use_init_image:
        # determine max frames from length of input frames
        anim_args.max_frames = len(inputfiles) - 1
        print(f"Using {anim_args.max_frames} input frames from {inputframes_path}...")

    # use first frame as init
    if anim_args.hybrid_composite != 'None' and anim_args.hybrid_use_first_frame_as_init_image:
        for f in inputfiles:
            args.init_image = str(f)
            args.use_init = True
            print(f"Using init_image from video: {args.init_image}")
            break

    return args, anim_args, inputfiles, cn_files, hybridframes_path

def get_resized_image_from_filename(filename, dimensions):
    img = cv2.imread(filename)
    return cv2.resize(img, dimensions, cv2.INTER_LANCZOS4)

def delete_all_imgs_in_folder(folder_path):
    files = list(pathlib.Path(folder_path).glob('*.jpg'))
    files.extend(list(pathlib.Path(folder_path).glob('*.png')))
    for f in files: os.remove(f)

