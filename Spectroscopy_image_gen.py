from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def generate_image(generator, latent_dim, label, color_mode):
    # Generate input noise
    noise = np.random.normal(0, 1, (1, latent_dim))
    
    # Convert label to numpy array and reshape 
    label = np.array([label]).reshape(-1, 1)
    
    # Generate image
    generated_image = generator.predict([noise, label])
    
    # Rescale the image from [-1,1] to [0,1]
    generated_image = 0.5 * generated_image + 0.5
    
    if color_mode == 'grayscale':
        generated_image = np.mean(generated_image, axis=-1)

    return generated_image[0] 

def display_image(image):
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')
    plt.show()

def main():

    generator_path = 'C:\\Users\\mfpsv\\Desktop\\Papers-ML e VOCs\\Testes\\DCGAN\\CGAN\\generator_model.h5'
    generator = load_model(generator_path)
    latent_dim = 200  

    i = 0
    
    # While cycle to iterate the menu
    while True:
        command = input("Enter 'generate' to create an image or 'g' to generate 10000 images, 'exit' to quit: ").strip().lower()
        if command == 'g':
            label_input = int(input("Enter the label for the image (e.g., 0 or 1): "))
            color_mode = input("Enter 'grayscale' for grayscale image, anything else for color: ").strip().lower()
            while i < 5000:
                image = generate_image(generator, latent_dim, label_input, color_mode)
                cv.imwrite('C:\\Users\\mfpsv\\PycharmProjects\\img_gen\\imagens_gen\\C' + str(i) + '.png', cv.blur(image, (5, 5)))
                i = i + 1

        if command == 'generate':
            label_input = int(input("Enter the label for the image (e.g., 0 or 1): "))
            color_mode = input("Enter 'grayscale' for grayscale image, anything else for color: ").strip().lower()
            image = generate_image(generator, latent_dim, label_input, color_mode)
            # display_image(cv.cvtColor(cv.GaussianBlur(image, (5, 5), 0), cv.COLOR_BGR2RGB))
            display_image(cv.cvtColor(cv.blur(image, (5, 5)), cv.COLOR_BGR2RGB))
            # display_image(cv.cvtColor(cv.medianBlur(image, 5), cv.COLOR_BGR2RGB))
            # display_image(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        elif command == 'exit':
            break
