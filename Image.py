"""
Including Libraries
"""
import csv
import cv2
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

"""
Defining Function 
"""

def Image_Visualization(Images):
    #Sort and Print Left, Center & Right Images
    center_image_name = img_path + Images[50][0].split('\\')[-1]
    center_image = mpimg.imread(center_image_name)
    flipped_center_image = np.fliplr(center_image)

    left_image_name = img_path + Images[50][1].split('\\')[-1]
    left_image = mpimg.imread(left_image_name)
    flipped_left_image = np.fliplr(left_image)

    right_image_name = img_path + Images[50][2].split('\\')[-1]
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
    plt.title('Flipped Right Camera Image')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.5)

    plt.savefig('Image_Visualization_Track_2.png')

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

Image_Visualization(sample_images)
