import os
import cv2
from modules.shared import opts

'''
⟫ Resume from timestring:
    • Resume requires 3 "actual" frames in order to work.
        • Regenerate the last full cycle since it is often corrupted by being interrupted 
    • Resume determines where last "actual" cadence frame was (index/turbo_steps-remainder), and then starts one cycle back from there.
        • You can always delete any number of frames going backwards and resume will correctly determine where to start, even if in the middle of a cadence cycle
        • Example: At turbo_steps 10, you could have 33 frames rendered (due to power failure or force quit of runtime)
            • Resume would start on frame index 30 (generation becomes 'turbo_next_image', the end of a cadence cycle)
            • Resume would use frame 20 as the history image that becomes 'turbo_prev_image' (it's the 'turbo_next_image' from the last cycle)
            • Cadence loop uses those to recreate the last cadence cycle, from frames 21 to 30
            • Note: Frame 0 is always output by itself first, so that the first cadence cycle from 1-10 can be created.
              • So, two cadence cycles worth of frames would actually be 21, not 20. We always need the end of last cycle.
              • This is why max_frames set to 120 yields 121 frames on disk. Frame 0, plus 12 cadence cycles of 10 frames.
    • So, with a turbo_steps of 10, you need at least 1 + (10 * 2) frames done in order to resume
        • The "actual" 3 frames are frame 0, 10, 20
'''

def get_resume_vars(folder, timestring, turbo_steps, img_history):
    DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)
    frame_count = 0
    for item in os.listdir(folder):
        # don't count txt files or mp4 files
        if ".txt" in item or ".mp4" in item:
            pass
        else:
            # Other image file types may be supported in the future.
            # We count through the files containing the timestring incrementally,
            # Excludes filenames containing 'depth' keyword (depth maps are saved in same folder)
            filename = item.split("_")
            if timestring in filename and "depth" not in filename:
                # only count sequentially numbered files, no gaps, starting at 000000000
                if format(frame_count,'09') in item:
                    frame_count += 1
                    # Show file list in dev debug mode
                    if DEBUG_MODE:
                        print(f"\033[36mResuming:\033[0m File: {'_'.join(filename)}")

    if frame_count == 0:
        raise IndexError(f"'Resume from timestring' might be enabled but there are no frames matching that timestring.\nCouldn't find '{folder}\{timestring}_000000000.png\n")

    print(f"\033[36mResuming:\033[0m {frame_count} frames counted, last 0-based frame index {frame_count-1}\n")

    # testing for frame 0 plus two cadence cycles (frame 20 at turbo_steps 10)
    if frame_count < 1 + turbo_steps * 2:
        raise IndexError("Not enough frames to resume. Need at least 2 times the diffusion_cadence.\n")

    # Correct for frame 0 and any number of extra frames at the end (so it can go back to an 'actual' frame)
    frame_idx = frame_count - (frame_count % turbo_steps)

    # move back one cadence cycle
    frame_idx = frame_idx - (turbo_steps * 2)
    print(f"\033[36mResuming:\033[0m Regenerating starting with frame index {frame_idx}")

    # history end range is one turbo step back from frame idx. end range is +1 because it's a range end, which is always +1 more than the end value of the range
    history_end_range = frame_idx + 1 - turbo_steps
    # history start range is whichever goes back furthest from the end_range, either back turbo_steps or back history's max states 
    history_start_range = min(history_end_range - turbo_steps, history_end_range - int(img_history.max_states))

    # set prev_img
    print(f"\033[36mResuming:\033[0m Populating prev_img variable from file index {frame_idx-turbo_steps}")
    prev_img = cv2.imread(os.path.join(folder, get_timestring_filename(timestring, frame_idx-turbo_steps)))

    # put last cadence cycle into img_history
    for this_idx in range(history_start_range, history_end_range):
        this_file = get_timestring_filename(timestring, this_idx)
        this_path = os.path.join(folder, this_file)
        print(f"Resume copying frame {this_idx} from {this_file} to image history.")
        img_history.add_state(cv2.imread(this_path), this_idx)

    return frame_idx, prev_img, img_history

def get_timestring_filename(timestring, frame_idx):
    return f"{timestring}_{frame_idx:09}.png"
