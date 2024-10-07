import tensorflow as tf
import tensorflow_datasets as tfds

# Load 'citrus' dataset
(train_data, test_data), info = tfds.load('citrus', split=['train', 'test'], with_info=True)

print(info.features)  # Get information about the dataset features
print(train_data.shape)  # Get the shape of the training data