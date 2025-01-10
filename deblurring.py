import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, LeakyReLU, BatchNormalization,Conv2DTranspose, Add, Activation, Flatten, Dense, InstanceNormalization, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2

def residual_block(x, filters):
    res = Conv2D(filters, kernel_size=3, strides=1, padding='same',
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    res = InstanceNormalization()(res)
    res = Activation('relu')(res)
    res = Conv2D(filters, kernel_size=3, strides=1, padding='same',
                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(res)
    res = InstanceNormalization()(res)
    return Add()([x, res])

def build_generator():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(64, kernel_size=7, strides=1, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(inputs)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    for _ in range(9):
        x = residual_block(x, 256)
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    outputs = Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh',
                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    return Model(inputs, outputs, name="Generator")

def build_discriminator():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(64, kernel_size=4, strides=2, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, kernel_size=4, strides=2, padding='same',
               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1)(x) 
    return Model(inputs, outputs, name="Discriminator")

def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (256, 256)) / 255.0  # Normalize to [0, 1]
            images.append(img)
    return np.array(images)

def train(generator, discriminator, blurred_images, clear_images, epochs, batch_size):
    optimizer_g = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
    optimizer_d = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(0, len(blurred_images), batch_size):
            blurred_batch = blurred_images[i:i + batch_size]
            clear_batch = clear_images[i:i + batch_size]
            fake_images = generator.predict(blurred_batch)
            real_labels = -np.ones((len(clear_batch), 1))  
            fake_labels = np.ones((len(fake_images), 1))  
            d_loss_real = discriminator.train_on_batch(clear_batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            misleading_labels = -np.ones((len(blurred_batch), 1))  
            g_loss = generator.train_on_batch(blurred_batch, misleading_labels)
            print(f"Batch {i // batch_size + 1}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")


def blur_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is not None:
                blurred = cv2.GaussianBlur(image, (3, 3), 0)
                output_path = os.path.join(output_folder, f"blurred_{filename}")
                cv2.imwrite(output_path, blurred)
                print(f"Blurred image saved: {output_path}")
            else:
                print(f"Failed to load image: {img_path}")
                
blurred_folder = "blurred_sketches"
clear_folder = "clear_sketches"
blur_images(clear_folder, blurred_folder)  # Create blurred images
blurred_images = load_images(blurred_folder)
clear_images = load_images(clear_folder)
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5, 0.999), loss=wasserstein_loss)
train(generator, discriminator, blurred_images, clear_images, epochs=500, batch_size=16)
