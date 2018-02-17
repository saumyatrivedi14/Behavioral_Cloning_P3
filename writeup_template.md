# **Behavioral Cloning** 
---
**Goals of the Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-nvidia.png "NVIDIA CNN Architecture"
[image2]: ./examples/Track_1_trial_1_Loss.png "Initial Mean Squared Error Loss for Track 1"
[image3]: ./examples/Track_1_trial_2_Loss.png "Final Mean Squared Error Loss for Track 1"
[image4]: ./examples/Image_Visualization_Track_1.png "Flipped Images of Track 1"
[image5]: ./examples/Image_Visualization_Track_2.png "Flipped Images of Track 2"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


**List of files included**

* Model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* Image.py - for generating figures for report and data visualization
* Model  - folder contains all the trained convolution neural network 
* Track_1_trial_1.mp4 - video of NN trained on data collected just from simulator  (Track 1)
* Track_1_trial_2.mp4 - video of NN trained with sample data provided (Track 1)
* Track_2.mp4 - video of NN trained on data collected just from simulator  (Track 2)
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Architecture

I have used a convolutional neural network similar to NVIDIA's CNN which was used by the team to train and drive autonomous cars. 

![alt text][image1]

The architecture consists of 9 layers, including a normalization layer, 5 convolutional layers (three 5x5 filter sizes and two 3x3 filter sizes kernels), and 3 fully connected layers with depths between 24 and 64 (refer Model.py). The network was build on Keras libraries with TensorFlow at backend. It consists of different size of input layers, flipping, cropping, normalization and Dropout layers to reduce overfitting and making the model more generic. The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. Keras Lambda layer was also used to add cropping layer, benefit of using cropping layer as a part of the network is so that it can use the GPU capability to crop multiple images simultaneously thus saving considerable time in training. 

#### 2. Reducing Overfitting

Data Normalization accompanied with Mean-Zeroing using the Keras lambda layer, followed by Cropping Layer top 50 rows (sky and mountains) and bottom 20 rows (Hood of car) which helps in reducing the train time. The model contains Dropout layers in order to reduce overfitting. Initially I started with only one Dropout layer after first Fully Connected Layer but it didn't reduce the overfitting, so I added one more layer of Dropout just after last Convolutional Layer. The keep_prob was kept at 40% for both the Dropout layers. I also trained the network on sample data provided by Udacity which helped in reducing overfitting. Below are the images of mean squared error losses before and after. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

##### Before

![alt text][image2]

##### After

![alt text][image3]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Training Strategy

The model was trained and validated on different data sets to avoid overfitting, test_train_split() function was used, with 80-20 split percentage, to split the training data for validation. I ran the simulator in training_mode and collected data but just for one lap total images were only 1418, training samples 1134 and validation samples 284 which was clearly not enough to train the network, So I trained it again on sample_data provided by Udacity (approximately 6000 images). 

As the car had three cameras, center, left and right, I flipped all three images, using numpy.fliplr() function, from these camera which helped the network to be more robust and generic as Track 1 is more biased towards left turns. below is the example of an instance were all three images from the camera are flipped and feed to the network. The steering angle was corrected for left and right camera training data, with too high steering correction factor, car starts to move in a zig-zag manner and too low steering correction would affect the cornering ability of the car in sharp turns, so 0.2 was selected by calibrating the correction factor.

![alt text][image4]

Video of the car doing a full lap in the simulator on Track 1 can be found in the directory, I collected two videos named trail 1 & 2  (Track_1_trial_1.mp4 & Track_1_trial_2.mp4)

### Optional Challenge (Track 2)

I ran the same model on Track 2 images and it did a pretty good job for almost 80% of the lap, only towards the end, it didn't clear a complete U-turn challenge. It still requires some fine tuning but I have attached the video of it in the directory. Flipped images of the camera on Track 2 are shown below and the video file name is Track_2.mp4.

![alt text][image5]

