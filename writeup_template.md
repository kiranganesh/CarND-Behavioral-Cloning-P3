
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

![Image](https://github.com/kiranganesh/CarND-Behavioral-Cloning-P3/blob/master/images/pic2.JPG)

##Results

With this relatively simple model, the car seems to do a good job (albeit going around a few curves that were "too close" for comfort) in going around Track1.

The model did not work well on Track2. One obvious enhancement that I need to make in the training is to account for the slopes and uneven horizons. This could be achieved by adding a few random rotational deltas to the image. I plan to tackle this as a separate challenge aside from the submission.

The main learning I have from this exercise is that developing models 


##Acknowledgements 

I learned a ton from David Silver's hands on session on youtube. Multiple comments on slack were extremely insightful as well. Its great to have a community like this to accelerate your learning.

