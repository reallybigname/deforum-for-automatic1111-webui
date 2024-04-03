import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class StyleTransfer:
    def __init__(self):
        # Load the pre-trained VGG19 model without the fully connected layers
        self.vgg = VGG19(weights='imagenet', include_top=False)
        # Get the intermediate layers representing style and content
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.content_layers = ['block5_conv2']
        self.style_outputs = [self.vgg.get_layer(name).output for name in self.style_layers]
        self.content_outputs = [self.vgg.get_layer(name).output for name in self.content_layers]
        # Create a model for extracting style and content features
        self.model = Model(inputs=self.vgg.input, outputs=self.style_outputs + self.content_outputs)
        self.model.trainable = False

    def preprocess_image(self, image):
        # Convert image to float32
        image = tf.cast(image, tf.float32)
        # Normalize image for VGG19 model
        image = preprocess_input(image)
        return image

    def deprocess_image(self, processed_image):
        # 'Un-normalize' the image
        x = processed_image.copy()
        # Perform the inverse of the preprocessing step
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result / num_locations

    def do(self, style_image, target_image, style_weight=1e-2, content_weight=1e4, num_iterations=200):
        # Preprocess style and target images
        style_image = self.preprocess_image(style_image)
        target_image = self.preprocess_image(target_image)

        # Pass the style and target images through the model
        style_features = self.model(style_image)
        content_features = self.model(target_image)

        # Compute the style loss
        style_loss = 0
        for style_target, style_combination in zip(style_features[:5], self.style_outputs):
            style_loss += tf.reduce_mean((self.gram_matrix(style_target) - self.gram_matrix(style_combination))**2)

        # Compute the content loss
        content_loss = tf.reduce_mean((content_features[5] - content_features[6])**2)

        # Total loss
        total_loss = style_weight * style_loss + content_weight * content_loss

        # Use Adam optimizer to minimize the total loss
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        train_step = opt.minimize(total_loss, var_list=[target_image])

        # Run optimization for a number of iterations
        for i in range(num_iterations):
            train_step.run()

        # Return the stylized target image
        return self.deprocess_image(target_image)

''' Usage:
    style_transfer = StyleTransfer()
    stylized_image = style_transfer.do(style_image, target_image)
'''