"""
Including Libraries
"""

import os
import csv
import cv2
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

"""
Defining Functions 
"""

def Image_Visualization(Images):
    #Sort and Print Left, Center & Right Images
    center_image_name = img_path + Images[0][0].split('/')[-1]
    center_image = mpimg.imread(center_image_name)
    flipped_center_image = np.fliplr(center_image)

    left_image_name = img_path + Images[0][1].split('/')[-1]
    left_image = mpimg.imread(left_image_name)
    flipped_left_image = np.fliplr(left_image)

    right_image_name = img_path + Images[0][2].split('/')[-1]
    right_image = mpimg.imread(right_image_name)
    flipped_right_image = np.fliplr(right_image)

    fig1 = plt.figure()

    plt.subplot(3,2,1)
    plt.imshow(left_image)
    plt.title('Left Camera Image')

    plt.subplot(3,2,2)
    plt.imshow(flipped_left_image)
    plt.title('Flipped Left Camera Image')

    plt.subplot(3,2,3)
    plt.imshow(center_image)
    plt.title('Center Camera Image')

    plt.subplot(3,2,4)
    plt.imshow(flipped_center_image)
    plt.title('Flipped Center Camera Image')

    plt.subplot(3,2,5)
    plt.imshow(right_image)
    plt.title('Right Camera Image')

    plt.subplot(3,2,6)
    plt.imshow(flipped_right_image)
    plt.title('Left Camera Image')

def Generator(samples, batch_size=32, path = './Track_2_Data/IMG/'):
    """
    Generate the images and steering angle for training
    `samples` is a list of pairs (`ImagePath`, `Steering Angles`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                C_imagePath = sample[0].split('/')[-1]
                L_imagePath = sample[1].split('/')[-1]
                R_imagePath = sample[2].split('/')[-1]
                C_angle = float(sample[3])

                C_Image = mpimg.imread(C_imagePath)
                #C_Image = cv2.cvtColor(C_Image, cv2.COLOR_BGR2RGB)
                images.append(C_Image)
                angles.append(C_angle)

                correction = 0.2 #steering correction to make it go straight
                L_angle = C_angle + correction
                R_angle = C_angle - correction

                L_Image = mpimg.imread(L_imagePath)
                #L_Image = cv2.cvtColor(L_Image, cv2.COLOR_BGR2RGB)
                images.append(L_Image)
                angles.append(L_angle)

                R_Image = mpimg.imread(R_imagePath)
                #R_Image = cv2.cvtColor(R_Image, cv2.COLOR_BGR2RGB)
                images.append(R_Image)
                angles.append(R_angle)            

            
                # Flipping
                Flipped_C_images = np.fliplr(C_Image)
                Flipped_C_angles = (-1.00*C_angle)
                images.append(Flipped_C_images)
                angles.append(Flipped_C_angles)

                Flipped_L_images = np.fliplr(L_Image)
                Flipped_L_angles = (-1.00*L_angle)
                images.append(Flipped_L_images)
                angles.append(Flipped_L_angles)

                Flipped_R_images = np.fliplr(R_Image)
                Flipped_R_angles = (-1.00*R_angle)
                images.append(Flipped_R_images)
                angles.append(Flipped_R_angles)

            # trim image to only see section with road
            images = np.array(images)
            steering_angle = np.array(angles)
            yield sklearn.utils.shuffle(images, steering_angle)

    

def Model():
    """NVIDIA Model"""
    keep_prob = 0.4

    #Pre-processing Layers
    model = Sequential()
    model.add(Lambda(lambda x: (x/255)-0.5, input_shape=(160,320,3))) #Normalizing Layer
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320))) #Cropping top 50 rows and bottom 20 rows

    #Convolutional Layers
    model.add(Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
    model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
    model.add(Convolution2D(48, 5, 5, activation='relu', border_mode='valid', subsample=(2,2)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))
    model.add(Dropout(keep_prob))

    #Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

"""
Workspace
"""
    
#Load and Initialize Data
img_path = "./Track_2_Data/IMG/"
csv_path = "./Track_2_Data/"
sample_images = []

with open(csv_path + "driving_log.csv",'r') as f:
    reader = csv.reader(f)
    next(reader, None) #skips the headers
    for line in reader:
        sample_images.append(line)
sklearn.utils.shuffle(sample_images) #Shuffle the data
print('Total Images: {}'.format(len(sample_images)))

#Splitting the images into training and validation sets
training, validation = train_test_split(sample_images, test_size=0.2)
print('Train Samples: {}'.format(len(training)))
print('Validation Samples: {}'.format(len(validation)))

#Generating and storing preprocessing data in memory
training_generator = Generator(training, batch_size=32, path = img_path)
validation_generator = Generator(validation, batch_size=32, path = img_path)

#defining Model
model = Model()
epochs = 5
    

#compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(training_generator, samples_per_epoch=len(training), validation_data = 
    validation_generator,
    nb_val_samples = len(validation), nb_epoch=epochs, verbose=1)

#Loss and Accuracy
print(history_object.history.keys())

#Saving the model
model.save('modelv2.h5')
print('Modelv2.h5 saved\n')

#Image_Visualization(sample_images)
