## MNIST Image Classification with TensorFlow

This project utilizes TensorFlow to build and train a model for classifying handwritten digits (0-9) from the MNIST dataset. Achieved a test accuracy of 98.5%.

**Project Structure:**

* `dataset_exploration.py`: Explores and visualizes the MNIST dataset.
* `image_classification.py`: Builds, trains, and evaluates a CNN for classifying MNIST digits.

**dataset_exploration.py**

This script performs the following tasks:

1. **Loads the MNIST dataset** using `tensorflow_datasets` with shuffling enabled to ensure random data access during training.
2. **Explores dataset features:** Extracts and saves information about the dataset features (image shape, data type, number of classes) to a text file named `exploratory_output.txt`. 
3. **Visualizes Sample Images:** Displays and saves 10 randomly chosen images with their corresponding labels as a PNG image named `mnist_sample_images.png`. 

**image_classification.py**

This script trains a Convolutional Neural Network (CNN) model for MNIST image classification:

1. **Loads MNIST data:** Loads the dataset using `tensorflow.keras.datasets.mnist`.
2. **Preprocesses data:**
    * Reshapes images to a format suitable for CNNs (28x28 pixels with one color channel).
    * Normalizes pixel values between 0 and 1 for better training performance.
    * One-hot encodes labels for multi-class classification (0 to 9).
3. **Defines the CNN model:**
    * Uses a sequential model with the following layers:
        * **Conv2D**: Applies a convolutional filter to extract features from the image.
        * **MaxPooling2D**: Reduces the spatial dimensionality of the data.
        * **Flatten**: Flattens the data into a single dimension for feeding to the dense layers.
        * **Dense**: Fully connected layer with ReLU activation for non-linearity.
        * **Dense**: Output layer with 10 units and softmax activation for probability distribution (one for each digit class).
4. **Compiles the model:**
    * Sets the optimizer ('adam') and loss function ('categorical_crossentropy') for training.
    * Defines accuracy as the evaluation metric.
5. **Trains the model:** Fits the model to the training data (`x_train` and `y_train`) for 10 epochs. Uses validation data (`x_test` and `y_test`) to monitor performance during training.
6. **Evaluates the model:** Evaluates the model's accuracy on the test set and prints the test accuracy. **Achieved a test accuracy of 98.5%.**

**Running the Project:**

1. Install all libraries found in `requirements.txt` (pip install -r requirements.txt).
2. Run `python dataset_exploration.py` to explore the data and save visualizations.
3. Run `python image_classification.py` to train and evaluate the model. The script will print the test accuracy.

**Future Work:** This is a basic implementation. Further improvements may include:

* Experimentation with different hyperparameters (learning rate, number of layers, etc.)
* Implementation of data augmentation techniques to improve model robustness.
* Exploration of more advanced CNN architectures (e.g., VGG16, ResNet).