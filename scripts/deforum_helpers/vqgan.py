import tensorflow as tf
import numpy as np
import io
from PIL import Image

class VQGAN:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = tf.saved_model.load(model_path)
        return model

    def generate_image(self, code):
        code_tensor = tf.convert_to_tensor(code, dtype=tf.float32)
        code_tensor = tf.expand_dims(code_tensor, 0)  # Add batch dimension
        generated_image = self.model(code_tensor)['default']
        generated_image = np.array(generated_image[0].numpy())
        return generated_image

    def encode_image(self, image):
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension
        codes = self.model.encode(image_tensor)
        return codes.numpy()[0]

    def decode_image(self, code):
        code_tensor = tf.convert_to_tensor(code, dtype=tf.float32)
        code_tensor = tf.expand_dims(code_tensor, 0)  # Add batch dimension
        decoded_image = self.model.decode(code_tensor)
        decoded_image = np.array(decoded_image[0].numpy())
        return decoded_image

    def encode_and_generate(self, image):
        code = self.encode_image(image)
        generated_image = self.generate_image(code)
        return code, generated_image

    def generate_and_decode(self, code):
        generated_image = self.generate_image(code)
        decoded_image = self.decode_image(code)
        return generated_image, decoded_image

    def save_image(self, image, file_path):
        pil_image = Image.fromarray(np.uint8(image))
        pil_image.save(file_path)

    def load_image(self, file_path):
        pil_image = Image.open(file_path)
        image = np.array(pil_image)
        return image

''' # Example usage of the VQGAN class

    # Instantiate VQGAN class with the path to the saved model
    vqgan = VQGAN(model_path='/path/to/saved_model')

    # Load an input image
    input_image = vqgan.load_image('input_image.jpg')

    # Encode the input image into a code
    code = vqgan.encode_image(input_image)

    # Generate an image based on the code
    generated_image = vqgan.generate_image(code)

    # Decode the code into an image
    decoded_image = vqgan.decode_image(code)

    # Save the generated image to a file
    vqgan.save_image(generated_image, 'generated_image.jpg')

    # Save the decoded image to a file
    vqgan.save_image(decoded_image, 'decoded_image.jpg')
'''
