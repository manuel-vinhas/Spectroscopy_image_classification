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
