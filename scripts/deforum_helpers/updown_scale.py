import cv2
import numpy as np
import torch
import torch.nn.functional as F
import copy
from .general_utils import debug_print

def updown_scale_whatevers(list_of_tuples):
    # Ex. list_of_tuples = [(img1, scale=2), (img2, shape=img2shape)]
    # returns shape, shape
    # Ex. list_of_tuples = [(img1, shape=shape1), (img2, shape=shape2)]
    # returns (img1downscaled, shape1), (img2downscaled, shape2)
    shape_return = []
    for whatever, shape, scale in list_of_tuples:
        shape_return.append(updown_scale_whatever(whatever, shape, scale))
    return shape_return

def updown_scale_whatever(whatever, shape=None, scale=None):
    '''
    Function: 'updown_scale' ins: whatever (img, flow, or depth), scale=N to upscale or shape=[height, width] to restore)
    • upscale in: tuple(whatever, scale=N) if scale not passed, scale=2 (2x)
             out: whatever_at_Nx, original_image_shape
    • do a warp function on whatever_at_Nx
    • downscale in: tuple(whatever_at_Nx, shape=original_image_shape) restores to original size
               out: new_whatever, whatever_at_Nx_shape (you'd normally discard whatever_at_Nx_shape)
    '''
    what = copy.deepcopy(whatever)
    bro = what.shape[:2]
  
    if isinstance(scale, str):
        scale = updown_scale_to_integer(scale)
  
    # exclusive or
    if (shape is not None) != (scale is not None):
        up = shape is None
        down = not up

        if down:
            scale = whatever.shape[0] / shape[0] 
                
        w = copy.deepcopy(what)
        a = {'scale': scale, 'warpterpolation': cv2.INTER_LANCZOS4}
        r = {'scale': 1/scale, 'warpterpolation': cv2.INTER_AREA}
        p = a if up else r
       
        i = round(w.shape[1] * p['scale'])
        t = round(w.shape[0] * p['scale'])

        if isinstance(w, torch.Tensor):
            you_are, awesome = 'map', 'depth'
        elif len(w.shape) == 2:
            if w.shape == (3, 3):
                you_are, awesome = 'matrix', 'perspective'
            elif w.shape == (2, 3):
                you_are, awesome = 'matrix', 'affine'
            else:
                you_are, awesome = 'image', 'grayscale'
        elif len(w.shape) == 3:
            if w.shape[2] == 2:
                you_are, awesome = 'flow', 'optical'
            else:
                you_are, awesome = 'image', 'color'

        # print(f"{'Downscaling' if down else 'Upscaling'} {awesome} {you_are} to {t}x{i} ")
          
        if you_are == 'map':
            dude = updown_scale_tensor(w, p['scale'], 'bilinear' if up else 'area')
        elif you_are == 'image':
            dude = cv2.resize(w, (i, t), interpolation=p['warpterpolation'])
        elif you_are == 'flow':
            dude = updown_scale_flow(w, (i, t), interpolation=p['warpterpolation'])
        elif you_are == 'matrix':
            dude = updown_scale_matrix(w, p['scale'])

        return dude, bro
    else:
        return what, bro

def updown_scale_to_integer(numx):
    # sssh
    s = str(numx)
    s = "1x" if s == "None" else s
    s = s.replace('x', '')
    h = float(s)
    return h

def updown_scale_tensor(input_tensor, scale, mode):    
    def determine_tensor_type(tensor):
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) == 3 and tensor.shape[1] > 1 and tensor.shape[2] > 1:
            return 'leres'
        elif isinstance(tensor, torch.Tensor) and len(tensor.shape) == 2:
            return '2d_tensor'
        elif isinstance(tensor, torch.Tensor) and len(tensor.shape) == 3 and tensor.shape[2] == 1:
            return 'midas'
        elif isinstance(tensor, torch.Tensor) and len(tensor.shape) == 3 and tensor.shape[2] == 2:
            return 'adabins'
        elif isinstance(tensor, torch.Tensor) and len(tensor.shape) == 3 and tensor.shape[2] == 3:
            return 'zoe'
        else:
            return 'unknown'

    tensor_type = determine_tensor_type(input_tensor)

    if tensor_type in ['midas', 'adabins', 'zoe', 'leres']:
        resized_tensor = F.interpolate(input_tensor.unsqueeze(0), scale_factor=scale, mode=mode)
        resized_tensor = resized_tensor.squeeze()
    elif tensor_type == '2d_tensor':
        # Handle 2D tensors
        resized_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode=mode)
        resized_tensor = resized_tensor.squeeze()
    else:
        raise ValueError("Unsupported tensor type")

    return resized_tensor

# used by updown_scale for optical flow resize
def updown_scale_flow(flow, dimensions, interpolation):
    width, height = map(float, dimensions)  # convert dimensions to float
    scaled_flow = cv2.resize(flow, (int(width), int(height)), interpolation=interpolation) * (width / flow.shape[1])
    return scaled_flow

def updown_scale_matrix(matrix, scale=None, shape=None):
    """
    Rescale a perspective or affine transformation matrix to work at a different resolution.

    Parameters:
    matrix (numpy.ndarray): The transformation matrix to be rescaled.
    scale (float, optional): The scale factor. If provided, the function will scale up the matrix.
    shape (tuple, optional): The original resolution as (width, height). If provided, the function will downscale the matrix.

    Returns:
    tuple: The rescaled transformation matrix and the original or scaled resolution.
    """
    is_affine = matrix.shape[0] == 2
    if scale is not None:
        # Scale up
        scale_matrix = np.array([[scale, 0], [0, scale]]) if is_affine else np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        rescaled_matrix = np.linalg.inv(scale_matrix) @ matrix
        original_resolution = (matrix[0, -1] / scale, matrix[1, -1] / scale)
        return rescaled_matrix, original_resolution
    elif shape is not None:
        # Scale down
        scale_matrix = np.array([[shape[0], 0], [0, shape[1]]]) if is_affine else np.array([[shape[0], 0, 0], [0, shape[1], 0], [0, 0, 1]])
        rescaled_matrix = scale_matrix @ matrix
        scaled_resolution = (matrix[0, -1] * shape[0], matrix[1, -1] * shape[1])
        return rescaled_matrix, scaled_resolution

def updown_scale_list(thing_man, shape=None, scale=None):
    get_in_shape = (thing_man[0].shape[1], thing_man[0].shape[0]) 
    if scale != 1:
        you_know = []
        for the in thing_man:
            if scale is not None:
                uptown, _ = updown_scale_whatever(the, scale=scale)
                you_know.append(uptown)
            elif shape is not None:
                downtown, _ = updown_scale_whatever(the, shape=shape)
                you_know.append(downtown)
        return you_know, get_in_shape
    else:
        return thing_man, get_in_shape

'''
The only legitimate reasons to use “updown_scale” are if you:
    • are / were thinking: “ I want to upscale a numpy image named 'blows' by 2x. ”
    • you also remembered: “ But I need to warp it with a custom function while it's upscaled! ”
    • and you're all like: “ Then, I want to downscale back to original size, even if I don't know what that was. ”

Example code:
    sucks, balls = updown_scale_whatever(blows)
    sucks = do_my_stupid_warp_function(sucks)
    blows, _ = updown_scale_whatever(sucks, shape=balls)

Example code in English:
    • send 'blows' image to updown_scale function
        • recieve 'sucks' image, the upscaled image of 'blows'
        • recieve 'balls' shape, the original shape of 'blows' for later
    • run “your” custom warp function on 'sucks', which is the upscaled 'blows' image
    • send 'sucks' in the shape of 'balls' to updown_scale function
        • 'blows' becomes your final downscaled version of 'sucks'
    • just discard '_' which is the shape of upscaled 'sucks' - you just don't need that trash in your life
'''
# 𝑤𝑎𝑟𝑝𝑡𝑒𝑟𝑝𝑜𝑙𝑎𝑡𝑖𝑜𝑛™️ • patent pending • all rights reserved
