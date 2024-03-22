import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization, Activation
import cv2 as cv
import os
from keras.callbacks import CSVLogger
from sklearn.utils import shuffle
import optuna
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold

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

def create_model(trial):

    # Define the model type
    model = Sequential()
    
    # Set the number of filters for the first convolutional layer
    log_num_filter = trial.suggest_int('log_num_filter_input', 2, 4)
    num_filter = 2 ** log_num_filter

    # Kernel dimensions for x and y axis
    f_size_x = trial.suggest_categorical('f_size_input', [3, 5, 7])
    #f_size_y = trial.suggest_categorical('f_size_input_y', [3, 5, 7, 11])

    # Input convolutional layer
    model.add(
        Conv2D(num_filter, (f_size_x, f_size_x), activation='relu', input_shape=(img_height, img_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional layers
    n_layers = trial.suggest_int('n_conv_layers', 1, 3)
    for i in range(n_layers):
        if i == 0:
            log_num_filter = trial.suggest_int(f'num_filter_conv_{i}', 3, 6)
            num_filter = 2 ** log_num_filter
        if i == 1:
            log_num_filter = trial.suggest_int(f'num_filter_conv_{i}', 5, 7)
            num_filter = 2 ** log_num_filter
        if i == 2:
            log_num_filter = trial.suggest_int(f'num_filter_conv_{i}', 7, 10)
            num_filter = 2 ** log_num_filter
        model.add(Conv2D(num_filter, (f_size_x, f_size_x), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    # Dense layers
    n_layers = trial.suggest_int('n_dense_layers', 1, 5)
    for j in range(n_layers):
        log_num_filter = trial.suggest_int(f'num_hidden_dense_{j}', 5, 7)
        num_hidden = 2 ** log_num_filter
        model.add(Dense(num_hidden, activation='relu'))

    # Dropout layer
    dropout_rate = trial.suggest_categorical('dropout_rate_conv', [0.2, 0.5])
    model.add(Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Set the learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

return model

def objective(trial):
    # Build the model
    model = create_model(trial)
    
    # Train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate the model
    y_pred = model.predict(test_images) > 0.5
    accuracy = accuracy_score(test_labels, y_pred)
return accuracy

#Uncomment to optimize cross validation model evaluation method
# def objective(trial):
#     # KFold cross-validation
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
#     accuracies = []
#     fold_count = 0
#     for train_index, val_index in kf.split(all_images, all_labels):
#         # Split data into training and validation sets
#         X_train, X_val = all_images[train_index], all_images[val_index]
#         y_train, y_val = all_labels[train_index], all_labels[val_index]
#         fold_count += 1
        
#         # Build the model
#         model = create_model(trial)
    
#         # Train the model
#         model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=0)

#         # Evaluate the model
#         loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
#         accuracies.append(accuracy)

#         # Print kFold results
#         print(f"Results for fold {fold_count}:")
#         print(f"Validation Accuracy: {accuracy}")
#         print(f"Validation Loss: {loss}")
#     # Average accuracy
#     average_accuracy = np.mean(accuracies)
# return average_accuracy

# Define the optimization process
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1300)

# Print optimization results
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
