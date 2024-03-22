import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization, Activation
import cv2 as cv
import os
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
from sklearn.utils import shuffle
from keras.optimizers import SGD
import keras
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import tensorflow as tf

# Image dimensions and number of channels
img_height = 180
img_width = 408
num_channels = 1  

# Numpy arrays for train images
train_images = np.zeros((137, img_height, img_width))
train_labels = np.zeros(137)

# Numpy arrays for test images
test_images = np.zeros((51, img_height, img_width))
test_labels = np.zeros(51)

# Numpy arrays for validation images
val_images = np.zeros((15, img_height, img_width))
val_labels = np.zeros(15)

count1 = 0
path = "B:\Downloads\Doentes"
for files1 in os.listdir(path):
    img = cv.imread(str(path) + "\\" + files1)
    
    # Convert images to grayscale and normalize to the interval [0,1]
    img_normalized = (cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0)  
    
    if count1 < 57:
        if count1 < 6:
            val_images[count1] = img_normalized
            val_labels[count1] = 1
            count1 = count1 + 1
            continue
        train_images[count1 - 6] = img_normalized
        train_labels[count1 - 6] = 1
    else:
        test_images[count1 - 57] = img_normalized
        test_labels[count1 - 57] = 1
    count1 = count1 + 1

count2 = 0
path2 = "B:\Downloads\Controlo"
for files2 in os.listdir(path2):
    img = cv.imread(str(path2) + "\\" + files2)
    
    # Convert images to grayscale and normalize to the interval [0,1]
    img_normalized = (cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0)
    
    if count2 < 95:
        if count2 < 9:
            val_images[count2 + 6] = img_normalized
            val_labels[count2 + 6] = 0
            count2 = count2 + 1
            continue
        train_images[count2 + 51 - 9] = img_normalized
        train_labels[count2 + 51 - 9] = 0
    else:
        test_images[count2 - 95 + 20] = img_normalized
        test_labels[count2 - 95 + 20] = 0
    count2 = count2 + 1

def create_model(optimizer="adam", neuronios=64, l1=4, l2=16, l3=64, l4=256, n2 = 128, f_size=3):
    
    # Define the model type
    model = Sequential()
    
    # First set of convolutional and pooling layers
    model.add(Conv2D(l1, (f_size, f_size), activation='relu', input_shape=(img_height, img_width, num_channels)))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Second set of convolutional and pooling layers
    model.add(Conv2D(l2, (f_size, f_size), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Third set of convolutional and pooling layers
    model.add(Conv2D(l3, (f_size, f_size), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Fourth set of convolutional and pooling layers
    model.add(Conv2D(l4, (f_size, f_size), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Set of fully-connected layers
    model.add(Flatten())
    model.add(Dense(neuronios, activation='relu'))
    model.add(Dense(n2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])  
return model

# Wrapper to make the keras model compatible with scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=76, verbose=2)

# Parameters values for the optimization
param_dist = {
    'optimizer': ['SGD', 'Adam', 'Adagrad', 'RMSprop'],
    'l1': [4, 8],
    'l2': [16, 32],
    'l3': [64, 128],
    'l4': [512, 1024],
    'neuronios': [32, 64, 128, 256, 512],
    'n2': [32, 64, 128, 256, 512],
    'f_size': [3, 5, 7],
    'epochs': [10, 15, 20],
    'batch_size': [32]
}

# Shuffle datasets
train_images, train_labels = shuffle(train_images, train_labels)

# Create a predefined split for cross-validation (-1 for train, 0 for test)
test_fold = [-1] * len(train_images) + [0] * len(test_images)  
ps = PredefinedSplit(test_fold)

# Random search definition
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10000, cv=ps, verbose=2)
random_search_result = random_search.fit(np.vstack((train_images, test_images)),
                                         np.hstack((train_labels, test_labels)))

# Convert the random search results to a Pandas DataFrame
results_df = pd.DataFrame(random_search_result.cv_results_)

# Save to CSV
results_df.to_csv('grid_search_results.csv', index=False)

# Save best model estimator
best_model = random_search_result.best_estimator_.model
best_model.save('best_model.Keras')

# Print the results
print("Best Score: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))

