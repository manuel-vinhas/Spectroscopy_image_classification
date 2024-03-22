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
import keras
from keras import regularizers
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold

# Numpy arrays for train images
train_images = np.zeros((152, img_height, img_width, 3))
train_labels = np.zeros(152)

# Numpy arrays for test images
test_images = np.zeros((51, img_height, img_width, 3))
test_labels = np.zeros(51)

# Image importation
count1 = 0
path = "B:\Downloads\Doentes"
for files1 in os.listdir(path):
    img = cv.imread(str(path) + "\\" + files1)
    
    # Pre-process images with the option 1 (CLAHE)
    img_processed = img_preprocess(img, 1)

    # If uncommented - convert images to grayscale and normalize to the interval [0,1]
    # img_normalized = (cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0)
    
    if count1 < 57:
        train_images[count1] = img_processed
        train_labels[count1] = 1
    else:
        test_images[count1 - 57] = img_processed
        test_labels[count1 - 57] = 1
    count1 = count1 + 1

count2 = 0
path2 = "B:\Downloads\Controlo"
for files2 in os.listdir(path2):
    img = cv.imread(str(path2) + "\\" + files2)
    
    # Pre-process images with the option 1 (CLAHE)
    img_processed = img_preprocess(img, 1)

    # If uncommented - convert images to grayscale and normalize to the interval [0,1]
    # img_normalized = (cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0)
    
    if count2 < 95:
        train_images[count2 + 57] = img_processed
        train_labels[count2 + 57] = 0
    else:
        test_images[count2 - 95 + 20] = img_processed
        test_labels[count2 - 95 + 20] = 0
    count2 = count2 + 1

# Image preprocessing
def img_preprocess(img, opt):
    # Convert the image from BGR to YCrCb
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # Divide the image into Y, Cr, and Cb
    y, cr, cb = cv.split(ycrcb_img)

    if opt == 0:
        # Equalize the histogram of the Y channel
        y_eq = cv.equalizeHist(y)
    else:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)

    # Merge the Y channel with Cr and Cb channels
    eq_ycrcb_img = cv.merge((y_eq, cr, cb))

    # Convert the YCrCb image to BGR
    eq_image = cv.cvtColor(eq_ycrcb_img, cv.COLOR_YCrCb2BGR)

    image = eq_image / 255

    return image

# L1 loss
def l1_reg_loss(y_true, y_pred):
    reg_loss = 0.
    for layer in model.layers:
        # If the layer has the attribute then it's using the regularizer
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            reg = layer.kernel_regularizer
            reg_loss += reg.l1 * tf.reduce_sum(tf.abs(layer.kernel))
    return reg_loss

# L2 loss
def l2_reg_loss(y_true, y_pred):
    reg_loss = 0.
    for layer in model.layers:
        # If the layer has the attribute then it's using the regularizer
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            reg = layer.kernel_regularizer
            reg_loss += reg.l2 * tf.reduce_sum(tf.square(layer.kernel))
    return reg_loss

# Stop the training process when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=9)

# Create a callback to reduce the learning rate when the validation loss has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=1, min_lr=0.00001)

def CNN_1_model():
     model = Sequential()
     model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(16, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(64, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(256, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(64, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
return model

def CNN_2_model():
     model = Sequential()
     model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(16, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(64, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(256, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(64, activation='relu'))
     model.add(Dense(128, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
return model

def random_search_model():
     model = Sequential()
     model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(AveragePooling2D(pool_size=(2, 2)))
     model.add(Conv2D(32, (3, 3), activation='relu'))
     model.add(AveragePooling2D(pool_size=(2, 2)))
     model.add(Conv2D(128, (3, 3), activation='relu'))
     model.add(AveragePooling2D(pool_size=(2, 2)))
     model.add(Conv2D(512, (3, 3), activation='relu'))
     model.add(AveragePooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(32, activation='relu'))
     model.add(Dense(64, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
return model

def bayes_opt_model():
     model = Sequential()
     model.add(Conv2D(8, (5, 5), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(16, (5, 5), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(64, activation='relu'))
     model.add(Dropout(0.2))
     model.add(Dense(1, activation='sigmoid'))
return model

def 5cv_model():
     model = Sequential()
     model.add(Conv2D(16, (7, 7), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(32, (7, 7), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(128, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
    return model

def 5scv_model():
     model = Sequential()
     model.add(Conv2D(16, (7, 7), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(64, (7, 7), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(128, activation='relu'))
     model.add(Dense(64, activation='relu'))
     model.add(Dense(128, activation='relu'))
     model.add(Dropout(0.2))
     model.add(Dense(1, activation='sigmoid'))
return model

def new_preprocess_model():   
     model = Sequential()
     model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(8, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Conv2D(64, (3, 3), activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2)))
     model.add(Flatten())
     model.add(Dense(32, activation='relu'))
     model.add(Dense(64, activation='relu'))
     model.add(Dense(64, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
return model

def cgan_model():
    model = Sequential()
    model.add(Conv2D(8, (5, 5), activation='relu', input_shape=(img_height, img_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
return model

def VGG11(input_shape=(180, 408, 3), num_classes=1):
    model = tf.keras.Sequential()

    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape,
                     kernel_regularizer=regularizers.l1(0.00001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

   
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l1(0.00001)))

    return model

def inception_module(x, filters):
    # 1x1 conv
    conv1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l1(0.000001))(x)

    # 3x3 conv
    conv3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l1(0.000001))(x)
    conv3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l1(0.000001))(conv3)

    # 5x5 conv
    conv5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu', kernel_regularizer=regularizers.l1(0.000001))(x)
    conv5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l1(0.000001))(conv5)

    # 3x3 max pooling
    pooling = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pooling = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(pooling)

    return concatenate([conv1, conv3, conv5, pooling], axis=-1)


def googlenet(input_shape, num_classes):
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', kernel_regularizer=regularizers.l1(0.000001))(input_img)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1(0.000001))(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1(0.000001))(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x, [64, 96, 128, 16, 32, 32])  # 3a
    x = inception_module(x, [128, 128, 192, 32, 96, 64])  # 3b
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])  # 4a
    x = inception_module(x, [160, 112, 224, 24, 64, 64])  # 4b
    x = inception_module(x, [128, 128, 256, 24, 64, 64])  # 4c
    x = inception_module(x, [112, 144, 288, 32, 64, 64])  # 4d
    x = inception_module(x, [256, 160, 320, 32, 128, 128])  # 4e
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x, [256, 160, 320, 32, 128, 128])  # 5a
    x = inception_module(x, [384, 192, 384, 48, 128, 128])  # 5b
    x = AveragePooling2D((5, 5), strides=(1, 1), padding='valid')(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l1(0.000001))(x)

    model = Model(inputs=input_img, outputs=x)
    return model

# Model compilation and fitting
# Call the model definition function
model = create_model()

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00145378),
              loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=0)

# Evaluate the model 
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_accuracy)
print("Test Loss: ", test_loss)

# Uncomment to cross validation
# # Cross-validation
# # Combine train and test datasets
# all_images = np.concatenate((train_images, test_images), axis=0)
# all_labels = np.concatenate((train_labels, test_labels), axis=0)

# # KFold cross-validation
# #kf = KFold(n_splits=5, shuffle=True)
# accuracies = []
# losses = []

# # KFold stratified cross-validation
# kf = StratifiedKFold(n_splits=5, shuffle=True)

#  fold_count = 0
#  for train_index, val_index in kf.split(all_images, all_labels):
#      fold_count += 1
#      # Split data into training and validation sets
#      X_train, X_val = all_images[train_index], all_images[val_index]
#      y_train, y_val = all_labels[train_index], all_labels[val_index]

#      model = create_model()

#      # Compile the model 
#      model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00145378),
#                    loss='binary_crossentropy', metrics=['accuracy'])

#      # Train the model
#      model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=0)

#      # Evaluate the model
#      loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

#      print(f"Results for fold {fold_count}:")
#      print(f"Validation Loss: {loss}")
#      print(f"Validation Accuracy: {accuracy}")

#      accuracies.append(accuracy)
#      losses.append(loss)

# # Average accuracy and loss
# average_accuracy = np.mean(accuracies)
# average_loss = np.mean(losses)
# print('Average Cross-Validation Accuracy:', average_accuracy)
# print('Average Cross-Validation Loss:', average_loss)

# Results determination
# Predict the labels of the test set
y_pred = model.predict(all_images)

# Convert the decimal results to the binary classification classes (0 or 1)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Create the tabels with classes and averages
print(classification_report(all_labels, y_pred_binary))

# Calculate the AUC
aucu = roc_auc_score(all_labels, y_pred)
print('AUC: %.2f' % aucu)

# Create the confusion matrix
cm = confusion_matrix(all_labels, y_pred_binary)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Obtain the ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Obtain the precision-recall curve
precision, recall, _ = precision_recall_curve(all_labels, y_pred)

# Plot the precision-recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()


index = 155

# ----------------CONV1------------------------

img = all_images[index] 
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Get model predictions
predictions = model.predict(img)
top_pred_idx = np.argmax(predictions[0])

# Get the gradients of the top predicted class
with tf.GradientTape() as tape:
    last_conv_layer = model.get_layer('conv2d_2') 
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(img)
    class_out = model_out[:, top_pred_idx]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Generate the heatmap
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)
heatmap = np.maximum(heatmap, 0)
heatmap = heatmap[0]

# Resize the heatmap to the original image size
desired_size = (img_width, img_height)

# Resize the heatmap
heatmap = cv.resize(heatmap, dsize=desired_size)

# Convert the heatmap to the color map
heatmap = np.uint8(255 * heatmap)
heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_VIRIDIS)

# Convert image to uint8
img = np.squeeze(img, axis=0)
img = np.uint8(img * 255)

# # Convert the grayscale image to a 3-channel BGR image
# img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Superimpose the heatmap on original image
superimposed_img = heatmap * 0.4 + img
superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

#Display the image
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
plt.imshow(heatmap)
plt.axis('off')
plt.show()
plt.imshow(cv.cvtColor(superimposed_img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
