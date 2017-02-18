
##Video Summary

Watch this video for the final outcome of this project! 

https://github.com/kiranganesh/CarND-Behavioral-Cloning-P3/blob/master/run2.mp4

##The Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

##Files

The main files for this submission are model.py, model.h5, run2.mp4 and writeup_template.md
```sh
model.py
model.h5
run2.mp4
writeup_report.md
```

Using the drive.py and the beta_simulator provided by Udacity, the car can be driven autonomously by executing
```sh
python drive.py model.h5
```
##Model Architecture 

I used the nVidia End-to-End Deep Learning Network for my model. (However, I kept the image sizes to be the same as what came out of the simulator and did not resize it to match what was published in the original nVidia model)

My Keras model looks like this:

```sh
   model = Sequential()
   model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
   model.add(Cropping2D(cropping=((70,25),(0,0)))) # 65 x 320

   model.add(Convolution2D(24,5,5,name='Conv1',subsample=(2,2),activation='relu'))
   model.add(Convolution2D(36,5,5,name='Conv2',subsample=(2,2),activation='relu'))
   model.add(Convolution2D(48,5,5,name='Conv3',subsample=(2,2),activation='relu'))
   model.add(Convolution2D(64,3,3,name='Conv4',subsample=(1,1),activation='relu'))
   model.add(Convolution2D(64,3,3,name='Conv5',subsample=(1,1),activation='relu'))
   
   model.add(Flatten())
   model.add(Dense(1164))
   model.add(Dropout(DROPOUT))
   model.add(Dense(100))
   model.add(Dropout(DROPOUT))
   model.add(Dense(50))
   model.add(Dropout(DROPOUT))
   model.add(Dense(10))
   model.add(Dense(1))
```

##Optimization

The model uses Adam optimizer (so no manual tuning of the learning rate) and MSE loss functions. The dropout layers were set to 10% to control overfitting. 

##Preprocessing - Images

I focused initially on building the simplest model that can handle Track1.

The training data was taken from the Udacity provided dataset. I also created my own custom dataset and tried separate runs with it. One thing I learned from this exercise was that your driving style can result in significantly different training sets (do ou consistently create small steering adjustments, or do you create infrequent, larger steering adjustments etc)

The images were cropped 70 pixels from the top, and 25 pixels from the bottom to keep just the main features of the road. I made a note of several other steps that one could do at this stage (color/RGB processing, grayscaling, brightness/contrast adjustments, resizing, rotations) I left the pictures as is. Only the center camera images were used

##Preprocessing - Steering Angles

A steering angle of 0.5 from the training run doesnt mean that a value of 0.49 or 0.52 is "incorrect" - probably they all can still do the required job done. So I added a random variation of +/- 10% to the steering angles. This makes the model more general and helps to reduce overfitting to precise values from the training set. 

##Data Augmentation - Images

Inorder to reduce the bias to left turns, all images were flipped (as well as their steering angles reversed) and added to the database. This doubles the number of data points in the training set and makes the data also more generalized.

##Data Normalization

After the above mentioned steps, the distribution of data looks like this

![Image](https://github.com/kiranganesh/CarND-Behavioral-Cloning-P3/blob/master/images/pic1.JPG)

Clearly, there is an over-concentration of values around the 0 angle of steering. This needs to be addressed and the data needs to be made more balanced. After reducing the overrepresented class the balanced data set looks like this:

![Image](https://github.com/kiranganesh/CarND-Behavioral-Cloning-P3/blob/master/images/pic1.JPG)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
