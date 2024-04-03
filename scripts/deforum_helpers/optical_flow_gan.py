import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class OpticalFlowGAN:
    def __init__(self, input_shape=(512, 512, 3), latent_dim=100):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        # Generator architecture
        model = models.Sequential()
        # Define generator layers (e.g., convolutional layers, upsampling layers)
        return model

    def build_discriminator(self):
        # Discriminator architecture
        model = models.Sequential()
        # Define discriminator layers (e.g., convolutional layers, pooling layers)
        return model

    def build_gan(self):
        # Combine generator and discriminator into a single model
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated_image = self.generator(gan_input)
        gan_output = self.discriminator(generated_image)
        gan = models.Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def train(self, image_pairs, epochs=100, batch_size=32):
        # Training loop
        for epoch in range(epochs):
            for batch in range(0, len(image_pairs), batch_size):
                # Select a batch of sequential image pairs
                batch_images = image_pairs[batch:batch + batch_size]
                real_images = batch_images[:, 0]  # First image in the pair
                target_images = batch_images[:, 1]  # Second image in the pair

                # Generate optical flows towards the second image
                generated_flows = self.generator.predict(real_images)

                # Train the discriminator
                real_labels = np.ones((batch_size, 1))
                fake_labels = np.zeros((batch_size, 1))
                d_loss_real = self.discriminator.train_on_batch(target_images, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(generated_flows, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator (via GAN)
                gan_input = np.random.randn(batch_size, self.latent_dim)
                gan_labels = np.ones((batch_size, 1))
                g_loss = self.gan.train_on_batch(gan_input, gan_labels)

                print(f'Epoch {epoch+1}/{epochs}, Batch {batch+1}/{len(image_pairs)}, D Loss: {d_loss}, G Loss: {g_loss}')

    def predict(self, image):
        # Generate optical flow prediction for a single image
        return self.generator.predict(image)
    
    def predict_opencv_flow(self, image):
        # Generate optical flow prediction for a single image
        generated_flow = self.generator.predict(image)
        
        # Convert the generated flow to OpenCV-compatible format
        flow_x = generated_flow[0, :, :, 0]  # Extract flow along x direction
        flow_y = generated_flow[0, :, :, 1]  # Extract flow along y direction
        opencv_flow = np.stack((flow_x, flow_y), axis=-1)
        
        return opencv_flow
