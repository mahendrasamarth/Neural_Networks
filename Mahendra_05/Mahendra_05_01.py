# Mahendra, Samarth
# 1001_974_557
# 2023_04_30
# Assignment_05_01


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_images(generated_images, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param generated_images:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        ax[i].imshow(generated_images[i, :, :], cmap='gray')
        ax[i].axis('off')
    return f, ax

class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate images from the generator model at the end of each epoch
    Uses the same noise vector to generate images at each epoch, so that the images can be compared over time
    """
    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("generated_images"):
            os.mkdir("generated_images")
        generated_images = self.generator(self.noise, training=False)
        generated_images = generated_images.numpy()
        generated_images = generated_images*127.5 + 127.5
        generated_images = generated_images.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(generated_images)
        plt.savefig(os.path.join("generated_images", f"generated_images_{epoch}.png"))
        # close the plot to free up memory
        plt.close()

def build_discriminator():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 16,kernel_size = (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model

def build_generator():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = 7*7*8, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape(target_shape = (7,7,8)))
    model.add(tf.keras.layers.Conv2DTranspose(filters=8,kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(5, 5), strides=(2,2), padding='same', use_bias=False,activation = 'tanh'))

    return model

class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = (real_loss + fake_loss)/2
        return total_loss
    
    def generator_loss(self,fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)


    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        # TRAINING CODE START HERE
        noise = tf.random.uniform((batch_size, 100))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise,training = True)
            real_output = self.discriminator(data,training = True)
            fake_output = self.discriminator(fake_images,training = True)
            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output,fake_output)


        desc_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        gen_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(desc_gradients, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {'d_loss':d_loss, 'g_loss':g_loss}

def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10]*127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")


    # build the discriminator and the generator
    discriminator = build_discriminator()
    generator = build_generator()


    # build the DCGAN
    dcgan = DCGAN(discriminator=discriminator, generator=generator)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, batch_size=64, callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    generated_images = generator(noise, training=False)
    plot_images(generated_images*127.5 + 127.5, 4, 4)
    plt.savefig("generated_images.png")

    generator.save('generator.h5')

if __name__ == "__main__":
    train_dcgan_mnist()