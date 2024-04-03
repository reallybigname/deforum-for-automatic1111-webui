import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from .image_functions import create_vertical_strip, create_horizontal_strip, write_text_on_image, bgr2gray
from .hybrid_flow import get_flow_from_images, image_transform_optical_flow

def conform_image(image, target, flow_factor, iterations=1, flow_method=None, raft_model=None, return_flow=False, return_iterations=False):
    image_new = np.copy(image)
    target_new = np.copy(target)

    # initialize iteration images list for debug
    if return_iterations:
        iteration_images = [image]

    # Adjust the easing curve to start larger and decrease gradually
    # easing_curve = lambda x: 1 - (1 - x) ** 3
    # Adjust the easing curve to start slower and finish faster
    easing_factors = [1 - (1 - (n + 1) / iterations) ** 3 for n in range(iterations)]
    # easing_factors = [easing_curve((n + 1) / iterations) for n in range(iterations)]

    # Loop over iterations
    for n in range(iterations):
        current_flow_factor = flow_factor * easing_factors[n]

        # Calculate optical flow between the current image and the target
        flow = get_flow_from_images(image_new, target_new, flow_method, raft_model)

        if return_flow:
            if n == 0:
                returned_flow = np.copy(flow * current_flow_factor)
            else:
                returned_flow = cv2.add(returned_flow, flow * current_flow_factor)

        # Apply the transformation with the current flow factor
        image_new = image_transform_optical_flow(image_new, flow, current_flow_factor)

        # Calculate and store difference for this iteration
        win_size = 11
        image_new_gray = bgr2gray(np.copy(image_new))
        target_new_gray = bgr2gray(np.copy(target_new))
        similarity = ssim(image_new_gray, target_new_gray, multichannel=False, win_size=win_size, channel_axis=-1, gaussian_weights=False, sigma=1.5)

        # modify flow factor based on difference, but ensure that the final step is always full flow, unmodified by ssim score
        # if n < iterations - 1:
        #     current_flow_factor *= (1 - similarity)

        # if returning iteration images, add images to iteration images, overlay text
        if return_iterations:
            image_with_text = write_text_on_image(image_new, f'Easing {easing_factors[n]:.3f} | Similarity: {similarity:.3f}')
            iteration_images.append(image_with_text)

    # returns dict
    r = {'image': image_new}
    if return_flow:
        r.update({'flow': returned_flow})
    if return_iterations:
        strip = create_horizontal_strip(iteration_images, max_width=3840, max_height=3840)
        r.update({'strip': strip})
    return r

def conform_image2(image, target, flow_factor, iterations=1, flow_method=None, raft_model=None, return_flow=False, return_iterations=False):
    image_new = np.copy(image)
    target_new = np.copy(target)

    # initialize iteration images list for debug
    if return_iterations:
        iteration_images = [image]

    # Adjust the easing curve to start larger and decrease gradually
    easing_curve = lambda x: 1 - (1 - x) ** 2
    easing_factors = [easing_curve((n + 1) / iterations) for n in range(iterations)]

    # Loop over iterations
    for n in range(iterations):
        current_flow_factor = flow_factor * easing_factors[n]

        # Calculate optical flow between the current image and the target
        flow = get_flow_from_images(image_new, target_new, flow_method, raft_model)

        if return_flow:
            if n == 0:
                returned_flow = np.copy(flow * current_flow_factor)
            else:
                returned_flow = cv2.add(returned_flow, flow * current_flow_factor)

        # Apply the transformation with the current flow factor
        image_new = image_transform_optical_flow(image_new, flow, current_flow_factor)

        # Calculate and store difference for this iteration
        win_size = 11
        image_new_gray = bgr2gray(np.copy(image_new))
        target_new_gray = bgr2gray(np.copy(target_new))
        similarity = ssim(image_new_gray, target_new_gray, multichannel=False, win_size=win_size, channel_axis=-1, gaussian_weights=False, sigma=1.5)

        # modify flow factor based on difference, but ensure that the final step is always full flow, unmodified by ssim score
        if n < iterations - 1:
            current_flow_factor *= (1 - similarity)

        # if returning iteration images, add images to iteration images, overlay text
        if return_iterations:
            image_with_text = write_text_on_image(image_new, f'Easing {easing_factors[n]:.3f} | Similarity: {similarity:.3f}')
            iteration_images.append(image_with_text)

    # returns dict
    r = {'image': image_new}
    if return_flow:
        r.update({'flow': returned_flow})
    if return_iterations:
        strip = create_horizontal_strip(iteration_images, max_width=3840, max_height=3840)
        r.update({'strip': strip})
    return r

def conform_images(img1, img2, alpha=1.0, beta=None, iterations=1, flow_method='None', raft_model=None, return_flow=False, return_flow2=False, return_iterations=False, return_iterations2=False):
    # return_flow returns both flows unless the return_flow2 is False (default)
    # return_iterations returns both sets of iterations unless the return_iteration2 is False (default)
    #                   only used for debugging
    # alpha is how much img1 should warp towards img2 (default alpha:1 - all the way)
    # beta  is how much img2 should warp towards img1 (beta is automatically calculated to balance if ommitted)
    # alpha and beta don't need to balance, but it is recommended
    if beta is None:
        beta = 1.0 - alpha

    kwargs = {
        'iterations': iterations,
        'flow_method': flow_method,
        'raft_model': raft_model
    }
    kwargs1 = {}
    kwargs2 = {}

    # set up kwarg dicts for the forward and reverse conform calls
    if return_flow:
        kwargs1.update({'return_flow': True})
        if return_flow2:
            kwargs2.update({'return_flow': True})        
    if return_iterations:
        kwargs1.update({'return_iterations': True})
        if return_iterations2:
            kwargs2.update({'return_iterations': True})        

    # add shared kwargs to kawrgs1 and kwargs2
    kwargs1.update(kwargs)
    kwargs2.update(kwargs)

    # get dicts from both conforms
    dict1 = conform_image(img1, img2, alpha, **kwargs1)
    dict2 = conform_image(img2, img1, beta,  **kwargs2)

    # returns dict
    r = {
        'image1': dict1['image'],
        'image2': dict2['image']
    }    
    if return_flow:
        r.update({'flow1': dict1['flow']})
        if return_flow2:
            r.update({'flow2': dict2['flow']})
    if return_iterations:
        img2_strip = create_horizontal_strip([img2]*(iterations+1), max_width=3840, max_height=3840)
        strip = [dict1['strip'], img2_strip]
        r.update({'strip1': create_vertical_strip(strip, max_width=3840, max_height=3840)})
        if return_iterations2:
            img1_strip = create_horizontal_strip([img1]*(iterations+1), max_width=3840, max_height=3840)
            strip = [dict2['strip'], img1_strip]
            r.update({'strip2': create_vertical_strip(strip, max_width=3840, max_height=3840)})
    return r
