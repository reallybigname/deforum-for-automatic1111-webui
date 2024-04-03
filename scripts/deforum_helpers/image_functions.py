import cv2
import numpy as np
import base64
import io
from PIL import Image

# image conversions PIL/np BGR/RGB all combos
def resize_pil_convert_to_bgr(img, dimensions):
    img_new = img.resize(dimensions, Image.LANCZOS)
    return cv2.cvtColor(np.asarray(img_new), cv2.COLOR_RGB2BGR)

def rgb_pil2np_bgr(rgb_img):
    return rgb2bgr(pil2np(rgb_img))

def bgr_np2pil_rgb(bgr):
    return np2pil(bgr2rgb(bgr))

def pil2np(pil):
    return np.array(pil)

def np2pil(np):
    return Image.fromarray(np)

def rgb2bgr(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def bgr2rgb(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
def bgr2gray_bgr(bgr_img):
    gray = bgr2gray(bgr_img)
    return gray2bgr(gray)

def bgr2gray(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

def gray2bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# various image functions

def create_img_tag(img):
    ''' returns html image tag with the image encoded as a base64 png '''
    _, buffer = cv2.imencode('.png', img)
    img_str = base64.b64encode(buffer).decode()
    img_tag = f'<img src="data:image/png;base64,{img_str}" />'
    return img_tag

def create_vertical_strip(images, max_height=None, max_width=None):
    ''' creates a vertical strip of images from a list '''
    # Resize images if specified
    if max_height is not None or max_width is not None:
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            if max_height is not None and h > max_height:
                scale_factor = max_height / h
                img = cv2.resize(img, (int(w * scale_factor), max_height))
                h, w = img.shape[:2]
            if max_width is not None and w > max_width:
                scale_factor = max_width / w
                img = cv2.resize(img, (max_width, int(h * scale_factor)))
            resized_images.append(img)
        images = resized_images
    
    # Calculate total height and maximum width
    total_height = sum(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    # Create a blank canvas for the strip
    strip = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    # Paste images onto the strip
    y_offset = 0
    for img in images:
        strip[y_offset:y_offset + img.shape[0], :img.shape[1]] = img
        y_offset += img.shape[0]

    return strip

def create_horizontal_strip(images, max_height=None, max_width=None):
    ''' creates a horizontal strip of images from a list '''
    # Resize images if specified
    if max_height is not None or max_width is not None:
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            if max_height is not None and h > max_height:
                scale_factor = max_height / h
                img = cv2.resize(img, (int(w * scale_factor), max_height))
                h, w = img.shape[:2]
            if max_width is not None and w > max_width:
                scale_factor = max_width / w
                img = cv2.resize(img, (max_width, int(h * scale_factor)))
            resized_images.append(img)
        images = resized_images
    
    # Calculate total width and maximum height
    total_width = sum(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)

    # Create a blank canvas for the strip
    strip = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Paste images onto the strip
    x_offset = 0
    for img in images:
        strip[:img.shape[0], x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1]

    return strip

def create_gif_pil(bgr_images, fps=1, loop=0):
    """ Create a GIF from a list of BGR images using PIL and encode it as base64. """
    try:
        pil_images = [Image.fromarray(img[..., ::-1]) for img in bgr_images]  # Convert BGR to RGB

        # Calculate frame duration in milliseconds
        duration_per_frame = int(1000 / fps)  # PIL expects duration in milliseconds

        # Create GIF in memory
        gif_data = io.BytesIO()
        
        # Ensure disposal method is set for all frames
        for img in pil_images:
            img.info['dispose'] = 2
        
        # Save GIF frames
        pil_images[0].save(gif_data, format='GIF', save_all=True, append_images=pil_images[1:], duration=duration_per_frame, loop=loop)
        gif_data.seek(0)

        # Encode as base64
        base64_gif = base64.b64encode(gif_data.getvalue()).decode('utf-8')
        return base64_gif
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None

def create_base64_gif_img_tag(bgr_images, fps=1, loop=0):
    ''' takes bgr images list, returns html tag with animated gif encoded as base64 '''
    base64_gif = create_gif_pil(bgr_images, fps=fps, loop=loop)
    return f'<img src="data:image/gif;base64,{base64_gif}" />'

def save_base64_gif_to_file(data, gif_path):
    """Save a base64-encoded GIF to a file."""
    try:
        gif_bytes = base64.b64decode(data)
        with open(gif_path, 'wb') as f:
            f.write(gif_bytes)
    except Exception as e:
        print(f"Error saving GIF to file: {e}")

def write_text_on_image(image, text, position=(25, 25), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.75, color=(255, 255, 255), shadow_color=(0, 0, 0), thickness=2, shadow_thickness=3):
    """ Writes text onto an image with a shadow. """
    i = np.copy(image)
    posx, posy = position[0], position[1]
    # create shadows with different shifts
    for shiftx, shifty in [[0, 0], [-1,-1], [-1, 0], [-1, 1]]:
        i = cv2.putText(i, text, (posx+shiftx, posy+shifty), font, font_scale, shadow_color, shadow_thickness)
    # draw text over the shadows
    i = cv2.putText(i, text, position, font, font_scale, color, thickness)
    return i
