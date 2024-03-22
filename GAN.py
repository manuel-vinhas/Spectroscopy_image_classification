from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, Multiply, Conv2DTranspose, Conv2D, LeakyReLU, Dropout
from keras.optimizers import Adam
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

def build_generator(latent_dim, num_classes, img_shape):
     noise = Input(shape=(latent_dim,))
     
     label = Input(shape=(1,), dtype='int32')
     label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

     model_input = Multiply()([noise, label_embedding])
     hidden = Dense(128, activation="relu")(model_input)
     hidden = Dense(128, activation="relu")(hidden)
     hidden = Dense(64, activation="relu")(hidden)
     hidden = Dense(32, activation="relu")(hidden)
     hidden = BatchNormalization(momentum=0.8)(hidden)
     hidden = Dense(np.prod(img_shape), activation="tanh")(hidden)
     img = Reshape(img_shape)(hidden)

     model = Model([noise, label], img)
     return model

def build_discriminator(img_shape, num_classes):
     img = Input(shape=img_shape)
     label = Input(shape=(1,), dtype='int32')
     label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
     label_embedding = Reshape(img_shape)(label_embedding)
    
     model_input = Multiply()([img, label_embedding])
     hidden = Flatten()(model_input)
     hidden = Dense(512, activation="relu")(hidden)
     hidden = Dense(256, activation="relu")(hidden)
     hidden = Dense(256, activation="relu")(hidden) 
     hidden = Dense(128, activation="relu")(hidden)
     hidden = Dense(128, activation="relu")(hidden)
     validity = Dense(1, activation="sigmoid")(hidden)
    
     model = Model([img, label], validity)
     return model

# Image parameters
img_rows = 180  
img_cols = 408  
channels = 3   
img_shape = (img_rows, img_cols, channels)

# Number of classes 
num_classes = 2

# Dimension of the latent space
latent_dim = 200  

# Optimizer definition
optimizer = Adam(0.0002, 0.5)
optimizer1 = Adam(0.000055, 0.5) 

# Build the discriminator
discriminator = build_discriminator(img_shape, num_classes)

# Compile the discriminator 
discriminator.compile(loss=['binary_crossentropy'],
                      optimizer=optimizer1,
                      metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, num_classes, img_shape)

# The generator receives noise and the target label as input
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])

# For the combined model, only the generator is trained
discriminator.trainable = False

# The discriminator receives a generated image as input and determines validity
valid = discriminator([img, label])

# The combined model (generator and discriminator)
# Trains the generator to dificult the discriminator classification
combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'],
                 optimizer=optimizer)

# Training hyperparameters
epochs = 7600
batch_size = 64
sample_interval = 200  

# Combine train and test datasets
X_train = np.concatenate((train_images, test_images), axis=0)
y_train = np.concatenate((train_labels, test_labels), axis=0)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):

    # Train discriminator
    
    # Select a random batch of images and corresponding labels
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs, labels = X_train[idx], y_train[idx]

    # Sample noise as generator input
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generate a batch of new images
    gen_imgs = generator.predict([noise, labels])

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    #  Train generator

    # Condition on labels
    sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

    # Train the generator so the discriminator labels the generator images as valid
    # On the combined model noise and the sampled_labels are the inputs of the generator. Valid array is used to set the output's target or label of the discriminator. Loss is then calculated. Which is used to update the generator weights
    g_loss = combined.train_on_batch([noise, sampled_labels], valid)

    # Print the generator and discriminator losses. And save generated images
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
        save_imgs(epoch, generator, latent_dim)
        
    if epoch == epochs-1:
        # Save the generator model
        generator.save('generator_model.h5')  # HDF5 file

        # Save the discriminator model
        discriminator.save('discriminator_model.h5')

        # Save the combined model
        combined.save('combined_model.h5')

  def save_imgs(epoch, generator, latent_dim, num_examples=10, dim=(1, 10), figsize=(10, 1)):
    # Sample noise
    noise = np.random.normal(0, 1, (num_examples, latent_dim))

    # Define labels for the generated images
    sampled_labels = np.arange(0, num_examples) % 2 

    # Generate images
    gen_imgs = generator.predict([noise, sampled_labels])

    # Rescale images [0,1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Plot generated images
    plt.figure(figsize=figsize)
    for i in range(gen_imgs.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        if gen_imgs.shape[-1] == 1:  # if grayscale
            plt.imshow(gen_imgs[i, :, :, 0], interpolation='nearest', cmap='gray')
        else:
            plt.imshow(cv.cvtColor(gen_imgs[i, :, :], cv.COLOR_BGR2RGB), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()

    # Save images to a directory
    plt.savefig("cgan_images/epoch_%d.png" % epoch)
    plt.close()
