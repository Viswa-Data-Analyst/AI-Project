#********************************************************Fashion Image Generation using GAN Approach******************************************************

# Library Improts
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D,BatchNormalization, Dense, LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
import numpy as np
from IPython.display import clear_output
import glob
import os
import cv2
from tqdm import tqdm
warnings.filterwarnings("ignore")


# Directory Setting
image_path = "dataset/"

# Get List of all images
images_file = glob.glob(image_path + '/**/*.jpg', recursive=True)
all_images = []
labels = []

# Array of Images
for file in tqdm(images_file):
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    all_images.append(img)
    labels.append(file.split('\\')[-2])

all_images = np.array(all_images)
labels = np.array(labels)

print(all_images.shape)
print(labels.shape)

#Images Plotting
def plot_images(images_array,labels):
    fig, axes=plt.subplots(2, 4, figsize=(20,14))
    axes= axes.flatten()
    i=0
    for img , ax in zip(images_array, axes):
        ax.imshow(img)
        ax.axis('off')
        title="Class Label ="+str(labels[i])
        ax.set_title(title)
        i+=1
    plt.tight_layout()
    plt.show()
        
plot_images(all_images[:10],labels[:10])

all_images = all_images.reshape(all_images.shape[0], 28*28)
print(all_images.min())
print(all_images.max())
all_images = (all_images.astype('float32') / 255 - 0.5) * 2
print(all_images.min())
print(all_images.max())

all_images.shape

# latent space dimension
latent_dim = 100

# images dimension 28x28
images_dim = 784

init = initializers.RandomNormal(stddev=0.02)

# Discriminator network creation
discriminator = Sequential()

# Input layer and hidden layers
discriminator.add(Dense(128, input_shape=(images_dim,), kernel_initializer=init))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(alpha=0.2))
# Output layer
discriminator.add(Dense(1, activation='sigmoid'))

# Discriminator Summary
discriminator.summary()

optimizer = Adam(lr=0.0002, beta_1=0.5)

discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
discriminator.trainable = False

# Generator network creation
generator = Sequential()

# Input layer and hidden layers
generator.add(Dense(128, input_shape=(latent_dim,), kernel_initializer=init))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(256))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
# Output layer 
generator.add(Dense(images_dim, activation='tanh'))

# Generator Summary
generator.summary()

Model = Sequential()
Model.add(generator)
Model.add(discriminator)
Model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

# Model Summary
Model.summary()

epochs = 500
batch_length = 16
smooth = 0.1

real_image = np.ones(shape=(batch_length, 1))
fake_image = np.zeros(shape=(batch_length, 1))

dis_loss = []
gen_loss = []

for e in range(epochs + 1):
    for i in range(len(all_images) // batch_length):

        # Train Discriminator weights
        discriminator.trainable = True

        # Real samples
        X_batch = all_images[i * batch_length : (i + 1) * batch_length]
        dis_loss_real = discriminator.train_on_batch(
            x=X_batch, y=real_image * (1 - smooth)
        )

        # Fake Samples
        z = np.random.normal(loc=0, scale=1, size=(batch_length, latent_dim))
        X_fake = generator.predict_on_batch(z)
        dis_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake_image)

        # Discriminator loss
        dis_loss_batch = 0.5 * (dis_loss_real[0] + dis_loss_fake[0])

        # Train Generator weights
        discriminator.trainable = False
        gen_loss_batch = Model.train_on_batch(x=z, y=real_image)

        print(
            "epoch = %d/%d, batch = %d/%d, discriminator_loss=%.3f, generator_loss=%.3f"
            % (
                e + 1,
                epochs,
                i,
                len(all_images) // batch_length,
                dis_loss_batch,
                gen_loss_batch[0],
            ),
            100 * " ",
            end="\r",
        )

    dis_loss.append(dis_loss_batch)
    gen_loss.append(gen_loss_batch[0])
    print(
        "epoch = %d/%d, discriminator_loss=%.3f, generator_loss=%.3f"
        % (e + 1, epochs, dis_loss[-1], gen_loss[-1]),
        100 * " ",
    )

    if e % 2 == 0:
        samples = 10
        x_fake = generator.predict(
            np.random.normal(loc=0, scale=1, size=(samples, latent_dim))
        )

        for k in range(samples):
            plt.subplot(2, 5, k + 1)
            plt.imshow(x_fake[k].reshape(28, 28), cmap="gray")
            plt.xticks([])
            plt.yticks([])

        
        plt.show()

#Metrics Plotting
plt.plot(dis_loss)
plt.plot(gen_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Discriminator', 'Adversarial'], loc='center right')
plt.show()

#References
#divyanshj16 - https://github.com/divyanshj16/GANs/blob/master/GANs-TensorFlow.ipynb
#Garima13a - https://github.com/Garima13a/MNIST_GAN/blob/master/MNIST_GAN_Solution.ipynb
#Kroosen - https://github.com/kroosen/GAN-in-keras-on-mnist/blob/master/GAN-keras-mnist-MLP.ipynb
