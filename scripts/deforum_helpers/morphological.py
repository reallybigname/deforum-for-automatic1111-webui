import cv2
import numpy as np
import random
import time
from .hybrid_flow import get_flow_from_images, image_transform_optical_flow
from .updown_scale import updown_scale_whatever
from .image_functions import create_img_tag, create_base64_gif_img_tag
from .general_utils import debug_print

def image_morphological_flow_transform(img, morpho_image_type, morpho_bitmap_threshold, flow_method, morpho_item, iterations, frame_idx, raft_model, flow_factor=1, updown_scale=1, return_image=True, return_flow=True, suppress_console=False):
    # extract colon separated vars 'operation[str]|kernel[str]|iterations[float]' from morpho schedule entry
    # iterations can be a float, but it gets rounded by extraction function to the nearest whole number

    # this mode only does morpho directly on image without using flow at all
    direct_mode = flow_method == 'No Flow (Direct)'

    # convert schedule item into op and kernel
    op, kernel = extract_morpho_schedule_item(morpho_item)
    
    # save original img shape, upscale img for morpho transform and flow calc
    if updown_scale > 1:
        img, img_shape = updown_scale_whatever(img, scale=updown_scale)

    # note: need to get op/kernel/iteration values '_used' because 'random' entries get translated to other codes
    if direct_mode:
        # no flow is used, 
        img_new, op_used, kernel_used, iterations_used = image_transform_morpho(img, morpho_image_type, morpho_bitmap_threshold, op, kernel, iterations, direct_mode=True)
    else:
        # get before/after image of morpho op in order to capture flow
        img_before_gray, img_after_gray, op_used, kernel_used, iterations_used = image_transform_morpho(img, morpho_image_type, morpho_bitmap_threshold, op, kernel, iterations)

        # get flow from before to after
        flow = get_flow_from_images(img_before_gray, img_after_gray, flow_method, raft_model)

        # warp image with flow - note: if not returning image, the image transform doesn't happen so flow factor is not applied here
        if return_image:
            img_new = image_transform_optical_flow(img, flow, flow_factor)
            
    # downscale image/flow for return
    if updown_scale > 1:
        if return_flow:  flow, _ = updown_scale_whatever(flow, shape=img_shape)
        if return_image: img_new, _ = updown_scale_whatever(img_new, shape=img_shape)
            
    oki_txt = f"Morphological transform: (op:{op_used}|kernel:{kernel_used}|iter:{iterations_used})"
    if direct_mode:
        console_text = f"{oki_txt} on frame {frame_idx} in {morpho_image_type} color mode"
    elif return_image:
        console_text = f"{oki_txt} on frame {frame_idx} using {flow_method} flow at factor {flow_factor:0.2f}"
    elif return_flow:
        console_text = f"{oki_txt} created {flow_method} flow for frame {frame_idx}"

    debug_print(console_text, not suppress_console)

    if direct_mode or (return_image and not return_flow):
        return img_new
    elif return_flow and not return_image:
        return flow
    else:
        return img_new, flow

def image_transform_morpho(img, morpho_image_type, morpho_bitmap_threshold, op, kernel, iterations=1, border=cv2.BORDER_DEFAULT, direct_mode=False):
    ''' performs morphological transformation with cv2.morphologyEx()
        returns grayscale bgr for optical flow calculations
    '''
    # convert on input to grayscale. If no valid mode, assumed to be color.
    # bitmaps are converted to grayscale first, and have threshold control
    # Note: with color images, each layer is processed separately by morphologyEx
    img_before = gray_or_bitmap(img, morpho_image_type, morpho_bitmap_threshold)

    # get codes for keys - returns the same op & kernel due to 'random' translating to a random key 
    operation, structuring, op, kernel = morpho_keys_to_codes(op, kernel)
  
    # do morphological transformation and store in image_after 
    img_after = cv2.morphologyEx(src=img_before, op=operation, kernel=structuring, iterations=iterations, borderType=border)

    # whether grayscale or bitmap, convert gray back to bgr grayscale
    if morpho_image_type in ['Grayscale', 'Bitmap']:
        img_before = cv2.cvtColor(img_before, cv2.COLOR_GRAY2BGR) 
        img_after = cv2.cvtColor(img_after, cv2.COLOR_GRAY2BGR)

    # returns names (not codes) for op/kernel console reporting from calling function
    if direct_mode:
        # direct mode uses no flow and only returns the image after
        return img_after, op, kernel, iterations 
    else:
        # returns images before and after
        return img_before, img_after, op, kernel, iterations


def gray_or_bitmap(img, image_type, bitmap_thresholds):
    """bitmap_thresholds must be formatted with a pipe low|high """
    # convert to grayscale, for bitmap too (before bitmap conversion)
    if image_type in ['Grayscale', 'Bitmap']:
        img_new = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)

        # convert to bitmap with low/high threshold
        if image_type == 'Bitmap':
            thresh = bitmap_thresholds.split('|')
            _, img_new = cv2.threshold(img_new, int(thresh[0]), int(thresh[1]), cv2.THRESH_BINARY)

        return img_new
    else:
        # any other image type is passed through
        return img

def morpho_keys_to_codes(op_key, kernel_key):
    # cv2 handles for operations/kernels in dicts
    operations = morpho_operations_get_dict()
    kernels =    morpho_kernels_get_dict()

    # 'if random, replace var with random choice from keys
    if op_key     == 'random': op_key =     random.choice(list(operations.keys()))
    if kernel_key == 'random': kernel_key = random.choice(list(kernels.keys()))
    
    # populate vars with morphological transformation codes from operations dict and structuring matrix from kernels dict
    operation = operations[op_key]
    structuring = kernels[kernel_key]
   
    # return op code, structuting code, op key and kernel key
    return operation, structuring, op_key, kernel_key

def morpho_get_random_op():
    operations = morpho_operations_get_dict()
    return random.choice(list(operations.keys()))

def morpho_get_random_kernel():
    kernels = morpho_kernels_get_dict()
    return random.choice(list(kernels.keys()))

def morpho_operations_get_dict():
    ''' returns dict of possible morphological operations
    '''
    return {
        'dilate': cv2.MORPH_DILATE,
        'erode': cv2.MORPH_ERODE,
        'open': cv2.MORPH_OPEN,         # same as dilate(erode(src,element))
        'close': cv2.MORPH_CLOSE,       # same as erode(dilate(src,element))
        'gradient': cv2.MORPH_GRADIENT, # same as dilate(src,element) − erode(src,element)
        'tophat': cv2.MORPH_TOPHAT,     # same as src − open(src,element)
        'blackhat': cv2.MORPH_BLACKHAT  # same as close(src,element) − src
    }

def get_morpho_operation_keys():
    all = morpho_operations_get_dict()
    return list(all.keys())

preset_kernels = {
    'rect': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    'cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
    'ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
}

def morpho_kernels_get_dict():
    ''' returns dict of possible morphological kernels
    '''
    custom_kernels = get_custom_kernels()
    all_kernels = {**preset_kernels, **custom_kernels}

    return all_kernels

def get_morpho_kernel_keys():
    all = morpho_kernels_get_dict()
    return list(all.keys())

def extract_morpho_schedule_item(morpho_item, separator='|', delimiter='+'):
    ''' extracts the 2 parts of the morphological operation schedule (str, str)
        trims whitespace, selects random items if colon separated list is given.
    '''
    p = morpho_item.split(separator)  # string split into 2 parts (op string, kernel string)
    sp = [s.strip() for s in p] # strip whitespace off of each part
    if len(p) == 2:
        op =     random.choice(sp[0].split(delimiter)) if delimiter in sp[0] else sp[0]
        kernel = random.choice(sp[1].split(delimiter)) if delimiter in sp[1] else sp[1]
        return op, kernel
    else:
        raise RuntimeError(f"Morphological schedule had a problem where there were more or less than 2 required parts on either side of the separator {separator}")

def get_custom_kernels():
    return {
        "hellipse": np.array([  [0,1,1,1,0],
                                [0,1,1,1,0],
                                [1,1,1,1,1],
                                [0,1,1,1,0],
                                [0,1,1,1,0]], dtype=np.uint8),

        "star": np.array([  [1,0,1,0,1],
                            [0,1,1,1,0],
                            [1,1,1,1,1],
                            [0,1,1,1,0],
                            [1,0,1,0,1]], dtype=np.uint8),

        "x": np.array([ [1,0,0,0,1],
                        [0,1,0,1,0],
                        [0,0,1,0,0],
                        [0,1,0,1,0],
                        [1,0,0,0,1]], dtype=np.uint8),

        "diamond": np.array([   [0,0,1,0,0],
                                [0,1,1,1,0],
                                [1,1,1,1,1],
                                [0,1,1,1,0],
                                [0,0,1,0,0]], dtype=np.uint8),

        "square": np.array([[1,1,1,1,1],
                            [1,0,0,0,1],
                            [1,0,0,0,1],
                            [1,0,0,0,1],
                            [1,1,1,1,1]], dtype=np.uint8),

        "checker": np.array([ [1,0,1,0,1],
                                [0,1,0,1,0],
                                [1,0,1,0,1],
                                [0,1,0,1,0],
                                [1,0,1,0,1]], dtype=np.uint8),

        "vline": np.array([ [0,0,1,0,0],
                            [0,0,1,0,0],
                            [0,0,1,0,0],
                            [0,0,1,0,0],
                            [0,0,1,0,0]], dtype=np.uint8),

        "hline": np.array([ [0,0,0,0,0],
                            [0,0,0,0,0],
                            [1,1,1,1,1],
                            [0,0,0,0,0],
                            [0,0,0,0,0]], dtype=np.uint8)                     
    }

'''
REFERENCE FOR EXISTING CV2 KERNELS:

cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
np.array([[1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1]], dtype=np.uint8)

# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
np.array([[0,0,1,0,0],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [0,0,1,0,0]], dtype=np.uint8)

# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
np.array([[0,0,1,0,0],
          [0,0,1,0,0],
          [1,1,1,1,1],
          [0,0,1,0,0],
          [0,0,1,0,0]], dtype=np.uint8)
'''

def get_morpho_kernel_keys_and_images():
    start_time = time.time()
    raw = []

    result = []
    for name, matrix in morpho_kernels_get_dict().items():
        img_tag, img = create_kernel_image_tag(matrix)
        raw.append(img)
        result.append(f"{name}<br>{img_tag}")
    
    # create animated gif from all that have been made and add it as the last entry
    result.append(f"random<br>{create_base64_gif_img_tag(raw, fps=4)}")
    
    debug_print(f"morpho key image list time: {time.time() - start_time:.2f} sec")

    return result

def get_morpho_op_keys_and_images():
    start_time = time.time()    
    result = []
    raw = []

    # reference image creation
    img = draw_morpho_op_sample()

    # operation none
    img_tag, img = create_op_image_tag('dilate', 'rect', its=0, img=img)
    result.append(f"none<br>{img_tag}")

    # operations
    for key, _ in morpho_operations_get_dict().items():
        img_tag, img = create_op_image_tag(key, 'rect', its=1, img=img)
        raw.append(img)
        result.append(f"{key}<br>{img_tag}")

    # create animated gif from all that have been made and add it as the last entry
    result.append(f"random<br>{create_base64_gif_img_tag(raw, fps=4)}")

    debug_print(f"morpho operation image list time: {time.time() - start_time:.2f} sec")   

    return result

def create_kernel_image_tag(matrix):
    matrix = np.where(matrix==1, 255, 0).astype(np.float32)
    resized_matrix = cv2.resize(matrix, (25, 25), interpolation=cv2.INTER_NEAREST)
    bgr_image = cv2.cvtColor(resized_matrix, cv2.COLOR_GRAY2BGR)

    return create_img_tag(resized_matrix), bgr_image.astype(np.uint8)

def create_op_image_tag(op='open', kernel='rect', its=1, img=None):
    if img is None:
        # size of op thumbnails
        img = draw_morpho_op_sample()
    
    if its > 0:
        img = image_morphological_flow_transform(img, '3-Color', '127|255', 'No Flow (Direct)', f'{op}|{kernel}', its, 'UI Rendering', None, flow_factor=1, return_flow=False, suppress_console=True)
    
    img_tag = create_img_tag(img)

    return img_tag, img

def draw_morpho_op_sample(letter=None, w=60, h=60, num_speckles=50):
    # Create a blank BGR image
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    if letter is None:
        letter = str(random.randint(10,99))

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Adjusted for visibility
    thickness = 2  # Adjusted for visibility
    colors = [(255, 255, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255)]  # White, Blue, Green, Red
    colors = reversed(colors)

    # Determine the centering of the text
    size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
    center_x = -7 + (w - size[0]) // 2
    center_y = (h + size[1]) // 2

    # Draw the letter four times, each in different color and shifted position
    for i, color in enumerate(colors):
        shift_x = center_x + i * 3
        shift_y = center_y
        cv2.putText(image, letter, (shift_x, shift_y), font, font_scale, color, thickness)

    # Add random speckles to the image
    for _ in range(num_speckles):
        speckle_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        speckle_x = random.randint(0, w - 1)
        speckle_y = random.randint(0, h - 1)
        cv2.circle(image, (speckle_x, speckle_y), random.randint(0,1), speckle_color, -1)  # -1 for filled circle

    return image
