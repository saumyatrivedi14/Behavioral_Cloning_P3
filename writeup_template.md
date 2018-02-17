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
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


My project includes the following files:
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

## Model Architecture and Training Strategy

### 1. Architecture

I have used a convolutional neural network similar to NVIDIA's CNN which was used by the team to train and drive autonomous cars. 

![alt text][image1]

The architecture consists of 9 layers, including a normalization layer, 5 convolutional layers (three 5x5 filter sizes and two 3x3 filter sizes kernels), and 3 fully connected layers with depths between 24 and 64 (refer Model.py). The network was build on Keras libraries with TensorFlow at backend. It consists of different size of input layers, flipping, cropping, normalization and Dropout layers to reduce overfitting and making the model more generic. The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. Keras Lambda layer was also used to add cropping layer, benefit of using cropping layer as a part of the network is so that it can use the GPU capability to crop multiple images simultaneously thus saving considerable time in training. 

### 2. Attempts to reduce overfitting in the model

The model contains Dropout layers in order to reduce overfitting. Initially I started with only one Dropout layer after first Fully Connected Layer but it didn't reduce the overfitting, so I added one more layer of Dropout just after last Convolutional Layer and it reduced overfitting. The keep_prob was kept at 40% for both the Dropout layers. I also trained the network on sample data provided by Udacity which helped in reducing overfitting. Below are the images of mean squared error losses before and after.

#### Before

![alt text][image2]

#### After

![alt text][image3]

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:


Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
