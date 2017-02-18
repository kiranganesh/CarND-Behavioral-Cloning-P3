import csv
import cv2
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

AWS = False
#AWS = True

def read_driving_log():
    
    lines = []

    with open('./udacitydata/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

def read_image(line):
    
    tokens = line.split('\\')
    filename = tokens[-1]
    local_path = "./udacitydata/" + filename
    image = cv2.imread(local_path)
    return image
   
def preprocess_data(lines):

    center_names = []
    left_names = []
    right_names = []
    measurements = []

    # skip the first line  
    for line in lines:
        center_names.append(line[0])
        left_names.append(line[1])
        right_names.append(line[2])
        measurements.append(float(line[3]))

    data = pd.DataFrame(measurements)
    data.columns = ['Angle']
    data['Index'] = data.index
    if (AWS == False):
        data['Angle'].hist(bins=30)
        plt.show()

    data_nonzero = data[data['Angle'] != 0.0]        
    data_zero = data[data['Angle'] == 0.0]        
    number = int(len(data_zero)/10)
    data_zero = data_zero.sample(number)
    result = pd.concat([data_zero, data_nonzero])
    if (AWS == False):
        result['Angle'].hist(bins=30)
        plt.show()
    
    return(result['Index'].values)
    
def read_driving_data(lines):
    
    images = []
    measurements = []
    # CORRECTION = 0.15
  
    for line in lines:

            measurement = float(line[3])

            # center image
            image = read_image(line[0])
            images.append(image)
            measurements.append(measurement)

            # left image
            # images.append(read_image(line[1]))
            # measurements.append(measurement+CORRECTION)

            # right image
            # images.append(read_image(line[2]))
            # measurements.append(measurement-CORRECTION)
            
    return images, measurements 

def view_datapoint(image, measurement):

    height = image.shape[0]
    width = image.shape[1]

    # Actual angle goes from -25 to +25. Normalized angle goes from -1 to +1
    angle = math.radians(measurement*25)
    
    y1 = int(height*0.9)
    y2 = int(height*0.5)
    x1 = int(width/2)
    x2 = int(x1 + math.tan(angle)*(y1-y2))

    cv2.line(image, (x1,y1),(x2,y2),(0,0,255),thickness=5)
    cv2.putText(image, str(measurement), org=(20,20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,0),thickness=2)

    plt.imshow(image)
    plt.show()

def augment_data_with_flips(images, measurements):      
    
    aug_images = []
    aug_measurements = []
    
    for image, measurement in zip(images, measurements):
        
        aug_images.append(image)
        aug_measurements.append(measurement)
        
        flipped_image = cv2.flip(image,1)
        flipped_measurement = float(measurement) * -1.0
                                   
        aug_images.append(flipped_image)
        aug_measurements.append(flipped_measurement)
    
    return aug_images, aug_measurements

def randomize_streeing_angles(measurements):
    
    for i in range(0, len(measurements)):
        measurements[i] = measurements[i] * (1 + random.uniform(0,0.1))

    return measurements
   
def evaluate_data(images, measurements):

    data = pd.DataFrame(measurements)
    if (AWS == False):
        data[0].hist(bins=30)
        plt.show()
    
def train_simple_model(X_train, y_train):

   model = Sequential()
   model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
   model.add(Flatten())
   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mse')
   model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
   model.save('model_simple.h5')

def train_nvidia_end2end_model(X_train, y_train):

   DROPOUT = 0.1

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
   
   model.compile(optimizer='adam', loss='mse')
   model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

   model.save('model.h5')

    
# read in all lines from driving log
lines = read_driving_log()

# downsample the 0 streeing angle cases
indices = preprocess_data(lines)
newlines = [lines[i] for i in indices]

# read data for the downsampled set
images, measurements = read_driving_data(newlines)

# augment image data with flips
aug_images = []
aug_measurements = []
aug_images, aug_measurements = augment_data_with_flips(images, measurements)      

# randomize steering angles 
aug_measurements = randomize_streeing_angles(aug_measurements)

# review final data
evaluate_data(aug_images, aug_measurements)

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)
print(X_train.shape)
print(y_train.shape)
    
#train_simple_model(X_train, y_train)

if (AWS == True):
    train_nvidia_end2end_model(X_train, y_train)
    exit()

# if __name__ == "__main__":
#    main()
