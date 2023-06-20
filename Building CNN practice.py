# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:33:23 2023

@author: tarun
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

##version of tensor flow
tf.__version__

##Prerocessing the training set
train_datagen = ImageDataGenerator(
                 rescale=1./255,
                 shear_range=0.2,
                 zoom_range=0.2,
                 horizontal_flip=True)  ##here an object is created for the purpose of data augumentation. Here the features will be changed that will add variablity in the trainingset and it will help
                                        ##the model from getting overfitted.
                                        ##filters used are rescale, shear_range, zoom_range,horizontal_flip

training_set = train_datagen.flow_from_directory(
                'dataset/training_set', #dataset for training
                target_size=(64,64),    ##output size of the image, larger size would take more time for triaining the data
                batch_size=32,          ##number of pcitures to be fed in a go
                class_mode='binary')
##now preprocess test set
##we will rescale it to 1./255 as the training set but we will not do any data augumentation since these images are new and will act like images in production in real world enviornment

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

##initialzing the CNN

cnn = tf.keras.models.Sequential()

##Step 1 convolution layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu', input_shape = [64,64,3]))  ##Conv2d belongs to the same class layers and it is 
                                                        #similiar to dense class that helps in creating layers. Filters
                                                        # will be number of filters requried, kernalz_size it shape of the matrix, so 3/3,
                                                        #activation is relu, imput shape shape should same as target size in training set.
                                                        #(64,64,3). Note 3 in the end is there because 3 means colored image. Had it been black and
                                                        #white it would be 1.
                                                        
# Max pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) ##pool size it size of the matrix for pooling, so 2/2.
                                                           ## and strides is movement, so 2 steps. Padding means
                                                           ##if during strides, it is not 2/2 in size....

##Add second convolution and pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu')) ##only change is that input_shape is removed because now we are at second layer and input is not
                                                                          ##coming from raw data that we have to resize or rescale it.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

##Flattening, converting the data into one dimentional vector
cnn.add(tf.keras.layers.Flatten())

##add fully connected layer
cnn.add(tf.keras.layers.Dense(units = 128, activation='relu')) ##all layers and neuron are connected. we have choosen 128 neurons in the hidden layer

##output layer
cnn.add(tf.keras.layers.Dense(units = 1, activation='sigmoid')) ##since output will be one, so one neuron

##we will now train the model
cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

##Training the CNN on the training set and evaluating it on the Test set
cnn.fit(x= training_set, validation_data= test_set, epochs= 25) ##training set will be validated on test set

##making a single  prediction
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size= (64,64)) ##we have uploaded a dogs image

##now we have to change it to PIL since predict method expects array as inputs.
test_image = image.img_to_array(test_image)

##we also have to define batach number since during training the data we did provide batch number
##so even though there is only one image, we still have to provide batch dimen so that algo recognizes it

test_image = np.expand_dims(test_image, axis=0) ##axis=0 meaning first dimenion added batch

result = cnn.predict(test_image/255.0)  ##by 255 I normalized the test image
    
training_set.class_indices
if result[0][0] > 0.5:  ##since result is in batch, with first 0, we are accessing 1st batch and within 1st batch, 0, we are accessing the first element
    prediction = 'dog'
else:
    prediction ='cat'

print(prediction)

##prediction is dog and it is correct





















