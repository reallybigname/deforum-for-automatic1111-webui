import os
import cv2
import pkg_resources
import numpy as np
from PIL import Image
from .load_images import load_image
from skimage.exposure import match_histograms
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from .general_utils import debug_print
from .image_functions import resize_pil_convert_to_bgr

class color_coherence:
    def __init__(self):
        self.matcher = ColorMatcher()

    def maintain_colors(self, image, sample_image, sample_alpha, mode, idx, cc_mix_outdir=None, timestring=None, suppress_console=False, console_msg=""):
        ''' main function for color_coherence, assumes BGR array for image and color match sample
        '''
        # ensure alpha value within normalized range
        sample_alpha = min(max(sample_alpha, 0), 1)

        # ensure bit depth for image and color sample
        matched = np.copy(image.astype(np.uint8))
        sample = np.copy(sample_image.astype(np.uint8))

        # blend image with sample for color coherence alpha
        # non-destructive of colors, since this function swaps individual pixels, rather than doing a blend operation
        sample = optimized_pixel_diffusion_blend(sample, matched, sample_alpha, cc_mix_outdir=cc_mix_outdir, timestring=timestring, idx=idx)

        skimage_version = pkg_resources.get_distribution('scikit-image').version
        is_skimage_v20_or_higher = pkg_resources.parse_version(skimage_version) >= pkg_resources.parse_version('0.20.0')
        match_histograms_kwargs = {'channel_axis': -1} if is_skimage_v20_or_higher else {'multichannel': True}
    
        if mode in ['HM', 'Reinhard', 'MVGD', 'MKL', 'HM-MVGD-HM', 'HM-MKL-HM']:
            # color-matcher by Christopher Hahne [GitHub:'hahnec']
            matched = self.matcher.transfer(src=image, ref=sample, method=mode.lower())
            matched = Normalizer(matched).uint8_norm()
        elif mode == 'RGB':
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            color_match = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2RGB)
            matched = match_histograms(image, color_match, **match_histograms_kwargs)
            matched = cv2.cvtColor(matched, cv2.COLOR_RGB2BGR)
        elif mode == 'HSV':
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
            color_match = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2HSV)
            matched = match_histograms(image, color_match, **match_histograms_kwargs)
            matched = cv2.cvtColor(matched, cv2.COLOR_HSV2BGR)
        else: # mode == 'LAB': (default)
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
            color_match = cv2.cvtColor(sample.astype(np.uint8), cv2.COLOR_BGR2LAB)
            matched = match_histograms(image, color_match, **match_histograms_kwargs)
            matched = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        
        # eventually make mechanism to log console msgs that are suppressed
        debug_print(f"Color coherence {mode} applied [sample {sample_alpha:.2f}/{1-sample_alpha:.2f} image] to frame {idx} | {console_msg} ", not suppress_console)

        return matched

def get_cc_sample_from_image_path(color_coherence_image_path, dimensions):
    cc_sample = load_image(color_coherence_image_path)
    cc_sample = resize_pil_convert_to_bgr(cc_sample, dimensions)
    print(f"Color match sample acquired from 'Image Path': {color_coherence_image_path}")
    return cc_sample

def get_cc_sample_from_video_frame(color_coherence_source, outdir, cc_vid_folder, frame_name, frame_idx, dimensions):
    # get new color coherence sample for video every frame
    cc_vid_frame_path = os.path.join(outdir, cc_vid_folder, frame_name + f"{frame_idx:09}.jpg")
    cc_sample = Image.open(cc_vid_frame_path)
    cc_sample = resize_pil_convert_to_bgr(cc_sample, dimensions)
    print(f"Color match sample acquired for {frame_idx} from {color_coherence_source}: {cc_vid_frame_path}")
    return cc_sample

def optimized_pixel_diffusion_blend(image1, image2, alpha, cc_mix_outdir=None, timestring=None, idx=None):
    alpha = min(max(alpha,0),1)
    beta = 1 - alpha
    
    # Create a random matrix the same shape as your image
    random_matrix = np.random.uniform(0, 1, image1.shape[:2])
    
    # Create a mask for where in the random matrix values are less than alpha
    alpha_mask = random_matrix < alpha
    
    # Create a mask for where in the random matrix values are less than beta
    beta_mask = (random_matrix >= alpha) & (random_matrix < alpha + beta)
    
    # Initialize result as a copy of image1
    result = np.copy(image1)
    
    # Apply the masks
    result[alpha_mask] = image1[alpha_mask]
    result[beta_mask] = image2[beta_mask]

    # if filepath is defined, save the result there
    if cc_mix_outdir is not None and timestring is not None and idx is not None:
        full_filepath = os.path.join(cc_mix_outdir, f'{timestring}_{idx:09}.jpg') 
        cv2.imwrite(full_filepath, result.astype(np.uint8))

    return result

