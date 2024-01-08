# Single image classification
Using Tensorflow vision files 
A Convolutional Neural Network (CNN) is a type of deep learning model designed for tasks involving images or grid-like data. 
CNNs have shown remarkable performance in tasks like image classification, object detection, and segmentation. 
They consist of multiple layers, including the input layer, convolutional layers, activation layers, pooling layers, fully connected layers, and the output layer. 
Let’s go through each layer in a typical CNN

The process involves
**Creating the Model**

To create the model, one simply calls the Create CNN method just coded that builds the multilayer Convolutional Neural Network

**Training the Model**

To train the model, we need to call the fit method and pass it the training data we loaded earlier, we’ll have it fit the training data 40 times to help the model converge to a decent accuracy

**Testing against Your own Data**

Lets create a method that allows us to take an image from our google drive and convert it to an array to be processed by our model
