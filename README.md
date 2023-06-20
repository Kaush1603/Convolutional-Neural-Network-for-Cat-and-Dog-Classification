# Convolutional-Neural-Network-for-Cat-and-Dog-Classification
This repository contains code for training a Convolutional Neural Network (CNN) model to classify images of cats and dogs. The model is trained on a dataset consisting of 4000 images of cats and 4000 images of dogs, with a total of 8000 training examples. The model achieves 89% accuracy on the training set and 79.78% validation accuracy on a separate test set consisting of 1000 cat images and 1000 dog images.

# Dataset
The dataset used for training the model contains a total of 8000 images, with 4000 cat images and 4000 dog images. The images are split into a training set and a test set with a 80:20 ratio (6400 training images and 1600 test images).

# Model Architecture
The CNN model architecture consists of two convolutional layers followed by max pooling layers, and a fully connected layer. The key details of the model architecture are as follows:

# Convolutional Layer 1:

Number of filters: 32
Filter size: 3x3
Activation function: ReLU
Max Pooling Layer 1:

Pool size: 2x2
Strides: 2

# Convolutional Layer 2:

Number of filters: 32
Filter size: 3x3
Activation function: ReLU

# Max Pooling Layer 2:

Pool size: 2x2
Strides: 2

# Fully Connected Layer:

Number of units: 128
Activation function: ReLU

# Output Layer:

Activation function: Sigmoid (Binary classification)

# Training
The model was trained using the Adam optimizer and binary cross-entropy loss. The training was performed for 25 epochs, with a batch size of 32. The optimizer adjusts the model's weights based on the gradients computed during the forward and backward propagation steps.

During training, the model achieved an accuracy of 89% on the training set and 79.78% accuracy on the test set.

# Usage
To use the trained model for prediction or further development, follow these steps:

Clone the repository: git clone https://github.com/your-username/your-repository.git
Install the required dependencies: pip install -r requirements.txt
Load the trained model: model = keras.models.load_model('path/to/saved/model.h5')
Prepare your input data (images) for prediction.
Use the loaded model to make predictions on the input data: predictions = model.predict(input_data)
Feel free to modify the code and experiment with different parameters to improve the model's performance.
