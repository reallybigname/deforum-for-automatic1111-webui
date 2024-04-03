'''     ʀe🅰️lly🅱️igname ௹ompositing Ⅿasks
    ᵐᵃᵈᵉ ᵖᵒˢˢⁱᵇˡᵉ ᵇʸ ᴮⁱⁿᵍ ᴬᴵ  ᶜʰᵃᵗᴳᴾᵀ  ᴳᵒᵒᵍˡᵉ ᴬᴵ   ''' 

import cv2
import numpy as np
from .conform import conform_images
from .general_utils import debug_print

def make_composite_with_conform(composite_type, new, old, alpha, flow_method, raft_model=None, conform=1, order='Forward', iterations=1, suppress_console=False):
    # if no composite type given, just return the new image to become the prev_img (old behavior)
    if composite_type.lower() == 'none':
        return new
    else:
        # in forward order, a is the new image, b is the old 
        fwd = order == 'Forward'
        a = new if fwd else old
        b = old if fwd else new

        # loop compositing only does a conforms using a flow method
        if flow_method != 'None':
            # conform images together by a conform factor
            # conform factor 1 makes A conform to B | 0.5 makes A and B conformed half-way towards each other | 0 will make B conformed to A
            conform_dict = conform_images(a, b, alpha, iterations=iterations, flow_method=flow_method, raft_model=raft_model)
            a = conform_dict['image1']
            b = conform_dict['image2']
            flow_msg = f"| {flow_method} flow conform factor {conform:.2f} using {iterations}"
        else:
            flow_msg = ''
       
        # console msg
        debug_print(f"Loop composite being created in '{order}' mode | filter: {composite_type} | alpha: {alpha:.2f}" + flow_msg, not suppress_console)

        # now that conform has completed (to hopefully bring the images into alignment), we composite the two images into a return variable
        return make_composite(composite_type, a, b, alpha)

# ௹all this function without any args to receive a dict with two list {'color': [list...], 'grayscale': [list...]}
# Ex. to access color/grayscale list: make_composite()['color'] | make_composite()['grayscale']
def make_composite(mask_type=None, a=None, b=None, alpha=1):  
    # this block of functions work with 3-channel color or 1-channel grayscale images
    def mask_type_blend(a, b):
        return a # blend only needs to return 'a' since all are blended with b later in function with alpha
    def mask_type_multiply(a, b):
        return a * b / 255
    def mask_type_screen(a, b):
        return 255 - ((255 - a) * (255 - b) / 255)
    def mask_type_overlay(a, b):
        return np.where(a <= 128, 2 * a * b / 255, 255 - 2 * (255 - a) * (255 - b) / 255)
    def mask_type_darken(a, b):
        return np.minimum(a, b)
    def mask_type_lighten(a, b):
        return np.maximum(a, b)
    def mask_type_dodge(a, b):
        return np.minimum(255, np.divide(a * 255, 255 - b + 1e-6, where=(b != 255)))
    def mask_type_burn(a, b):
        return 255 - np.minimum(255, np.divide((255 - a) * 255, b + 1e-6, where=(b != 0)))
    def mask_type_soft_light(a, b):
        return ((1 - (a / 255)) * b) + ((a / 255) * (255 - (255 - b) * (255 - a) / 255))
    def mask_type_hard_light(a, b):
        return np.where(a <= 128, mask_type_multiply(a, b), mask_type_screen(a, b))
    def mask_type_difference(a, b):
        return cv2.absdiff(a, b)
    def mask_type_exclusion(a, b):
        return a + b - 2 * a * b / 255
    def mask_type_addition(a, b):
        return np.clip(a + b, 0, 255)
    def mask_type_subtraction(a, b):
        return np.clip(a - b, 0, 255)
    def mask_type_color_dodge(a, b):
        return np.clip((a * 255) / (255 - b + 1e-6), 0, 255)
    def mask_type_color_burn(a, b):
        return 255 - np.clip((255 - a) * 255 / (b + 1e-6), 0, 255)
    def mask_type_vivid_light(a, b):
        return np.where(b <= 128, 255 - np.clip((255 - a) * 255 / (2 * b + 1e-6), 0, 255), np.clip((a * 255) / (2 * (255 - b) + 1e-6), 0, 255))
    def mask_type_linear_light(a, b):
        return np.clip(a + 2 * b - 255, 0, 255)
    def mask_type_dissolve(a, b, alpha=0.5):
        mask = np.random.binomial(1, alpha, a.shape)
        return a * mask + b * (1 - mask)
    def mask_type_luma_lighten_only(a, b):
        a_ycc, b_ycc = (bgr2ycrcb(image) for image in (a, b))
        a_y, a_cr, a_cb = cv2.split(a_ycc)
        b_y, b_cr, b_cb = cv2.split(b_ycc)
        result_y = np.where(a_y > b_y, a_y, b_y)
        result = cv2.merge((result_y, a_cr, a_cb))
        return ycrcb2bgr(result.astype(np.uint8))
    def mask_type_luma_darken_only(a, b):
        a_ycc, b_ycc = (bgr2ycrcb(image) for image in (a, b))
        a_y, a_cr, a_cb = cv2.split(a_ycc)
        b_y, b_cr, b_cb = cv2.split(b_ycc)
        result_y = np.where(a_y < b_y, a_y, b_y)
        result = cv2.merge((result_y, a_cr, a_cb))
        return ycrcb2bgr(result.astype(np.uint8))
    def mask_type_grain_extract(a, b):
        return np.clip(a - b + 128, 0, 255)
    def mask_type_grain_merge(a, b):
        return np.clip(a + b - 128, 0, 255)
    def mask_type_divide(a, b):
        return np.clip((a / (b + 1e-6)) * 255, 0, 255)
    
    # this block of functions only work with color images
    def mask_type_hue(a, b):
        a_hsv, b_hsv = (bgr2hsv(image) for image in (a, b))
        combined_hsv = cv2.merge([b_hsv[:, :, 0], a_hsv[:, :, 1], a_hsv[:, :, 2]])
        return hsv2bgr(combined_hsv)
    def mask_type_saturation(a, b):
        a_hsv, b_hsv = (bgr2hsv(image) for image in (a, b))
        combined_hsv = cv2.merge([a_hsv[:, :, 0], b_hsv[:, :, 1], a_hsv[:, :, 2]])
        return hsv2bgr(combined_hsv)
    def mask_type_hsv_hue(a, b):
        a_hsv, b_hsv = (bgr2hsv(image) for image in (a, b))
        combined_hsv = cv2.merge([b_hsv[:, :, 0], a_hsv[:, :, 1], a_hsv[:, :, 2]])
        return hsv2bgr(combined_hsv)
    def mask_type_hsv_saturation(a, b):
        a_hsv, b_hsv = (bgr2hsv(image) for image in (a, b))
        combined_hsv = cv2.merge([a_hsv[:, :, 0], b_hsv[:, :, 1], a_hsv[:, :, 2]])
        return hsv2bgr(combined_hsv)
    def mask_type_hsl_color(a, b):
        a_hsl, b_hsl = (bgr2hls(image) for image in (a, b))
        combined_hsl = cv2.merge([b_hsl[:, :, 0], b_hsl[:, :, 1], a_hsl[:, :, 2]])
        return hls2bgr(combined_hsl)
    def mask_type_hsv_value(a, b):
        a_hsv, b_hsv = (bgr2hsv(image) for image in (a, b))
        combined_hsv = cv2.merge([a_hsv[:, :, 0], a_hsv[:, :, 1], b_hsv[:, :, 2]])
        return hsv2bgr(combined_hsv)

    # dicts for function calls and for retrieval of lists of masks for color and grayscale
    mask_types_color_or_grayscale = {
        'blend': mask_type_blend,
        'addition': mask_type_addition,
        'subtraction': mask_type_subtraction,
        'vivid_light': mask_type_vivid_light,
        'linear_light': mask_type_linear_light,
        'multiply': mask_type_multiply,
        'screen': mask_type_screen,
        'overlay': mask_type_overlay,
        'darken': mask_type_darken,
        'lighten': mask_type_lighten,
        'soft_light': mask_type_soft_light,
        'hard_light': mask_type_hard_light,
        'dodge': mask_type_dodge,
        'burn': mask_type_burn,
        'color_dodge': mask_type_color_dodge,
        'color_burn': mask_type_color_burn,
        'luma_lighten_only': mask_type_luma_lighten_only,
        'luma_darken_only': mask_type_luma_darken_only,
        'grain_extract': mask_type_grain_extract,
        'grain_merge': mask_type_grain_merge,
        'dissolve': mask_type_dissolve,
        'difference': mask_type_difference,
        'exclusion': mask_type_exclusion,
        'divide': mask_type_divide
    }
    mask_types_color = {
        'hue': mask_type_hue,
        'saturation': mask_type_saturation,
        'hsv_hue': mask_type_hsv_hue,
        'hsv_saturation': mask_type_hsv_saturation,
        'hsl_color': mask_type_hsl_color,
        'hsv_value': mask_type_hsv_value
    }
    mask_types = {**mask_types_color_or_grayscale, **mask_types_color}
    
    if mask_type is not None:
        mask_type = mask_type.lower()    
    
    # make mask
    if mask_type in mask_types:
        # Remove alpha channel if present in either image
        a, b = [img[:, :, :3] if img.shape[-1] == 4 else img for img in [a, b]]
        
        # no need to composite with alpha 0
        if alpha == 0:
            return a
        else:            
            # ensures that grayscale images can't be selected for color filters by switching the lists
            mask_switcher = mask_types if a.ndim == 3 else mask_types_color_or_grayscale

            # Apply mask type function for each channel if BGR, else directly
            result = mask_switcher[mask_type](a, b)

            # Apply alpha/beta weighted blend between 'result' of a/b composite and 'a'
            result = composite_images_with_alpha(result, a, alpha)

            # Conditional normalization to the range [0, 255]
            if result.min() < 0 or result.max() > 255:
                result = ((result - result.min()) * (255 / (result.max() - result.min()))).astype(np.uint8)

            debug_print(f"Generated composite a/b mask source using: {mask_type} | alpha {alpha:.2f}")

            return result.astype(np.uint8)

    # if no arguments given, returns a list of available mask types from dict
    elif mask_type is None:
        return {
            'color': list(mask_types.keys()),
            'grayscale': list(mask_types_color_or_grayscale.keys())
        }

def composite_images_with_alpha(i1, i2, alpha, gamma=0.0):
    return cv2.addWeighted(i1.astype(np.uint8), alpha, i2.astype(np.uint8), 1-alpha, gamma)

def composite_images_with_mask(img1, img2, mask):
    # Normalize the mask image to have values between 0 and 1
    mask = mask / 255.0
    # Create a 3-channel version of the mask by stacking it
    mask = np.dstack([mask, mask, mask])
    # Composite the images
    composite = img1 * mask + img2 * (1 - mask)
    # Convert the composite image back to 8-bit
    composite = cv2.convertScaleAbs(composite)
    return composite

def bgr2hsv(image): return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
def bgr2hls(image): return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
def bgr2lab(image): return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
def bgr2ycrcb(image): return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
def hsv2bgr(image): return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
def hls2bgr(image): return cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
def lab2bgr(image): return cv2.cvtColor(image, cv2.COLOR_Lab2BGR)
def ycrcb2bgr(image): return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

