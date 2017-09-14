import csv
import cv2
import numpy as np

data_folder = 'a/'
correction = 0.1

def get_image(original_path):
    '''Load the image from specified path'''
    filename = original_path.split('/')[-1]
    current_path = data_folder + 'IMG/' + filename
    image = cv2.imread(current_path)
    return image

# Load the CSV file with recorded data
lines = []
with open(data_folder + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Load images for all 3 cameras and add flipped images as well
images = []
measurements = []
for line in lines:
    center = get_image(line[0])
    images.append(center)
    images.append(cv2.flip(center, 1))
    left = get_image(line[1])
    images.append(left)
    images.append(cv2.flip(left, 1))
    right = get_image(line[2])
    images.append(right)
    images.append(cv2.flip(right, 1))
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement * -1.0)
    measurements.append(measurement + correction)
    measurements.append((measurement + correction) * -1.0)
    measurements.append(measurement - correction)
    measurements.append((measurement - correction) * -1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Activation, Cropping2D, Reshape

# Build network model
model = Sequential()
# Normalize input
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Crop the input, we don't need the whole image
model.add(Cropping2D(cropping=((75,20), (0,0))))

# Follow the NVIDIA architecture
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

# Train the model
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
