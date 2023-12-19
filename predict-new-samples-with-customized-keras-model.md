---
title: "How to predict new samples with your TensorFlow / Keras model?"
date: "2020-02-21"
categories:
  - "buffer"
  - "deep-learning"
  - "frameworks"
tags:
  - "keras"
  - "machine-learning"
  - "model"
  - "neural-network"
  - "neural-networks"
  - "predict"
---


**Update 03/Nov/2020:** fixed textual error.


# File path
filepath = './path_to_model'

# Load the model
model = load_model(filepath, compile = True)

# A few random samples
use_samples = [5, 38, 3939, 27389]
samples_to_predict = []

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)
```

* * *



```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from extra_keras_datasets import emnist

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load EMNIST dataset
(input_train, target_train), (input_test, target_test) = emnist.load_data(type='digits')

# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Cast numbers to float32
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Scale data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

* * *

## Saving and loading the model

First, add the `save_model` and `load_model` definitions to our imports - replace the line where you import `Sequential` with:

```python
from tensorflow.keras.models import Sequential, save_model, load_model
```

Then, create a folder in the folder where your `keras-predictions.py` file is stored. Make sure to name this folder `saved_model` or, if you name it differently, change the code accordingly - because you next add this at the end of your model file:

```python
# Save the model
filepath = './saved_model'
save_model(model, filepath)
```

In line with how saving Keras models works, it saves the `model` instance at the `filepath` (i.e. that folder) that you specified.

Hooray! We now saved our trained model 🎉

### Loading

Loading the model for future usage is really easy - it's a two-line addition:

```python
# Load the model
model = load_model(filepath, compile = True)
```

* * *

## Generating predictions

With a loaded model, it's time to show you how to generate predictions with your Keras model! :)

Firstly, let's add Matplotlib to our imports - which allows us to generate visualizations. Then, also add Numpy, for number processing:

```python
import matplotlib.pyplot as plt
import numpy as np
```

```python
# A few random samples
use_samples = [5, 38, 3939, 27389]

# Generate plots for samples
for sample in use_samples:
  # Generate a plot
  reshaped_image = input_train[sample].reshape((img_width, img_height))
  plt.imshow(reshaped_image)
  plt.show()
```

Here they are:

- [![](images/dig_4.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_4.png)
    
- [![](images/dig_2.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_2.png)
    
- [![](images/dig_3.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_3.png)
    
- [![](images/dig_1.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_1.png)
    

We then extend this code so that we can actually store the samples temporarily for prediction later:

```python
# A few random samples
use_samples = [5, 38, 3939, 27389]
samples_to_predict = []

# Generate plots for samples
for sample in use_samples:
  # Generate a plot
  reshaped_image = input_train[sample].reshape((img_width, img_height))
  plt.imshow(reshaped_image)
  plt.show()
  # Add sample to array for prediction
  samples_to_predict.append(input_train[sample])
```

Then, before feeding them to the model, we convert our list into a Numpy array. This allows us to compute shape and allows Keras to handle the data more smoothly:

```python
# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
print(samples_to_predict.shape)
```

The output of the `print` statement: `(4, 28, 28, 1)`.

The next step is to generate the predictions:

```python
# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)
```

The output here seems to be a bit jibberish at first:

```python
[[8.66183618e-05 1.06925681e-05 1.40683464e-04 4.31487868e-09
  7.31811961e-05 6.07917445e-06 9.99673367e-01 7.10965661e-11
  9.43153464e-06 1.98050812e-10]
 [6.35617238e-04 9.08200348e-10 3.23482091e-05 4.98994159e-05
  7.29685112e-08 4.77315152e-05 4.25152575e-06 4.23201502e-10
  9.98981178e-01 2.48882337e-04]
 [9.99738038e-01 3.85520025e-07 1.05982785e-04 1.47284098e-07
  5.99268958e-07 2.26216093e-06 1.17733900e-04 2.74483864e-05
  3.30203284e-06 4.03360673e-06]
 [3.42538192e-06 2.30619257e-09 1.29460409e-06 7.04832928e-06
  2.71432992e-08 1.95419183e-03 9.96945918e-01 1.80040043e-12
  1.08795590e-03 1.78136176e-07]]
```

```python
# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)
```

This outputs `[6 8 0 6]`. Yeah! ✅ 🎉

- [![](images/dig_4.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_4.png)
    
- [![](images/dig_2.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_2.png)
    
- [![](images/dig_3.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_3.png)
    
- [![](images/dig_1.png)](https://www.machinecurve.com/wp-content/uploads/2020/02/dig_1.png)
    

## Full code

If you're interested, you can find the code as a whole here:

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import numpy as np

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load EMNIST dataset
(input_train, target_train), (input_test, target_test) = emnist.load_data(type='digits')

# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Cast numbers to float32
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Scale data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# # Save the model
filepath = './saved_model'
save_model(model, filepath)

# Load the model
model = load_model(filepath, compile = True)

# A few random samples
use_samples = [5, 38, 3939, 27389]
samples_to_predict = []

# Generate plots for samples
for sample in use_samples:
  # Generate a plot
  reshaped_image = input_train[sample].reshape((img_width, img_height))
  plt.imshow(reshaped_image)
  plt.show()
  # Add sample to array for prediction
  samples_to_predict.append(input_train[sample])

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)
print(samples_to_predict.shape)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)

# Generate arg maxes for predictions
classes = np.argmax(predictions, axis = 1)
print(classes)
```

* * *

## Summary

- Load EMNIST digits from the [Extra Keras Datasets](https://www.machinecurve.com/index.php/2020/01/10/making-more-datasets-available-for-keras/) module.
- Prepare the data.
- Define and train a Convolutional Neural Network for classification.
- Save the model.
- Load the model.
- Generate new predictions with the loaded model and validate that they are correct.
